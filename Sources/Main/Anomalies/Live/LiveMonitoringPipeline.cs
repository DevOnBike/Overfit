// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Alerting;
using DevOnBike.Overfit.Anomalies.Alerting.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using GptTrainingConfig = DevOnBike.Overfit.Anomalies.Training.GptTrainingConfig;

namespace DevOnBike.Overfit.Anomalies.Live
{
    /// <summary>
    /// Production live monitoring pipeline.
    ///
    /// Flow (every ScrapeInterval):
    ///   Prometheus → MetricSnapshot per pod → GptAnomalyDetector.Score → AlertEngine
    ///
    /// Usage:
    ///   await using var pipeline = LiveMonitoringPipeline.Create(
    ///       config: new LiveMonitoringOptions { ... },
    ///       checkpointPath: "checkpoint.bin",
    ///       trainingConfig: OfflineTrainingConfig.Quick,
    ///       new TeamsAlertSink(webhookUrl));
    ///
    ///   await pipeline.RunAsync(cancellationToken);
    /// </summary>
    public sealed class LiveMonitoringPipeline : IAsyncDisposable
    {
        private readonly PrometheusMetricSource _source;
        private readonly AlertEngine _alertEngine;
        private readonly GPT1Model _model;
        private readonly Dictionary<string, Gpt.GptAnomalyDetector> _detectors = new();
        private readonly LiveMonitoringOptions _options;
        private readonly GptTrainingConfig _trainingConfig;
        private bool _disposed;

        private LiveMonitoringPipeline(
            PrometheusMetricSource source,
            GPT1Model model,
            AlertEngine alertEngine,
            LiveMonitoringOptions options,
            GptTrainingConfig trainingConfig)
        {
            _source         = source;
            _model          = model;
            _alertEngine    = alertEngine;
            _options        = options;
            _trainingConfig = trainingConfig;
        }

        /// <summary>Creates a live monitoring pipeline from a trained checkpoint.</summary>
        public static LiveMonitoringPipeline Create(
            LiveMonitoringOptions options,
            GptTrainingConfig trainingConfig,
            string checkpointPath,
            params Alerting.Abstractions.IAlertSink[] sinks)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(trainingConfig);
            ArgumentNullException.ThrowIfNull(checkpointPath);

            if (!File.Exists(checkpointPath))
            {
                throw new FileNotFoundException($"GPT checkpoint not found: {checkpointPath}");
            }

            // Load GPT model from checkpoint
            var gptConfig = new GPT1Config
            {
                VocabSize     = Gpt.MetricTokenizer.VocabSize,
                ContextLength = trainingConfig.ContextLength,
                DModel        = trainingConfig.DModel,
                NHeads        = trainingConfig.NHeads,
                NLayers       = trainingConfig.NLayers,
                DFF           = trainingConfig.DModel * 4,
                TieWeights    = false,
                PreLayerNorm  = true,
            };

            var model = new GPT1Model(gptConfig);
            model.Eval();

            using var fs = File.OpenRead(checkpointPath);
            using var br = new BinaryReader(fs);
            model.Load(br);

            // Prometheus source
            var sourceConfig = new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = options.PrometheusBaseUrl,
                PodRegex          = options.PodRegex,
                ScrapeInterval    = options.ScrapeInterval,
            };
            var source = new PrometheusMetricSource(sourceConfig);

            // Alert engine
            var alertConfig = new AlertEngineConfig
            {
                AlertThreshold    = options.AlertThreshold,
                CriticalThreshold = options.CriticalThreshold,
                CooldownDuration  = options.CooldownDuration,
            };
            var alertEngine = new AlertEngine(alertConfig, sinks);

            return new LiveMonitoringPipeline(source, model, alertEngine, options, trainingConfig);
        }

        /// <summary>
        /// Runs the monitoring loop until cancellation.
        /// Scrapes Prometheus every ScrapeInterval, scores each pod, fires alerts.
        /// </summary>
        public async Task RunAsync(CancellationToken ct = default)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var scrapeTime = DateTime.UtcNow;
                    var series     = await _source.ReadAsync(ct).ConfigureAwait(false);
                    var snapshots  = ConvertToSnapshots(series, scrapeTime);

                    foreach (var snapshot in snapshots)
                    {
                        var detector = GetOrCreateDetector(snapshot.PodName);
                        var result   = detector.Score(snapshot);

                        if (result.IsWarmup)
                        {
                            continue;
                        }

                        _alertEngine.TryAlert(
                            result.PodName,
                            anomalyScore: result.Score,
                            reconstructionMse: result.Score);

                        _options.OnScore?.Invoke(result);
                    }
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _options.OnError?.Invoke(ex);
                }

                await Task.Delay(_options.ScrapeInterval, ct).ConfigureAwait(false);
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            foreach (var d in _detectors.Values)
            {
                d.Dispose();
            }
            _detectors.Clear();
            _source.Dispose();
            _model.Dispose();
            await _alertEngine.DisposeAsync().ConfigureAwait(false);
        }

        // ── Private ──────────────────────────────────────────────────────────

        private Gpt.GptAnomalyDetector GetOrCreateDetector(string podName)
        {
            if (_detectors.TryGetValue(podName, out var existing))
            {
                return existing;
            }
            var handle   = SlmRuntimeFactory.CreateGpt1(_model, SlmRuntimeMode.Cached);
            var detector = new Gpt.GptAnomalyDetector(handle, _options.ContextSnapshots);
            _detectors[podName] = detector;
            return detector;
        }

        private static List<MetricSnapshot> ConvertToSnapshots(
            IReadOnlyList<RawMetricSeries> series, DateTime timestamp)
        {
            // Group by pod name, accumulate feature values
            var podFeatures = new Dictionary<string, float[]>();

            foreach (var s in series)
            {
                var podName = s.Pod.PodName;
                if (!podFeatures.TryGetValue(podName, out var features))
                {
                    features = new float[MetricSnapshot.FeatureCount];
                    podFeatures[podName] = features;
                }

                var idx = MetricTypeIdToFeatureIndex(s.MetricTypeId);
                if (idx >= 0 && s.Samples.Count > 0)
                {
                    features[idx] = s.Samples[^1].Value;
                }
            }

            var result = new List<MetricSnapshot>(podFeatures.Count);
            foreach (var (podName, f) in podFeatures)
            {
                result.Add(new MetricSnapshot
                {
                    Timestamp             = timestamp,
                    PodName               = podName,
                    CpuUsageRatio         = f[0],
                    CpuThrottleRatio      = f[1],
                    MemoryWorkingSetBytes = f[2],
                    OomEventsRate         = f[3],
                    LatencyP50Ms          = f[4],
                    LatencyP95Ms          = f[5],
                    LatencyP99Ms          = f[6],
                    RequestsPerSecond     = f[7],
                    ErrorRate             = f[8],
                    GcGen2HeapBytes       = f[9],
                    GcPauseRatio          = f[10],
                    ThreadPoolQueueLength = f[11],
                });
            }
            return result;
        }

        // Maps PrometheusMetricSource MetricTypeId → MetricSnapshot feature index
        // MetricTypeId values must match what PrometheusMetricSource assigns.
        private static int MetricTypeIdToFeatureIndex(byte id) => id switch
        {
            0 => 0,  // CpuUsageRatio
            1 => 1,  // CpuThrottleRatio
            2 => 2,  // MemoryWorkingSetBytes
            3 => 3,  // OomEventsRate
            4 => 4,  // LatencyP50Ms
            5 => 5,  // LatencyP95Ms
            6 => 6,  // LatencyP99Ms
            7 => 7,  // RequestsPerSecond
            8 => 8,  // ErrorRate
            9 => 9,  // GcGen2HeapBytes
            10 => 10, // GcPauseRatio
            11 => 11, // ThreadPoolQueueLength
            _ => -1
        };
    }

    /// <summary>Options for LiveMonitoringPipeline.</summary>
    public sealed class LiveMonitoringOptions
    {
        /// <summary>Prometheus HTTP API base URL, e.g. "http://prometheus:9090".</summary>
        public required string PrometheusBaseUrl { get; init; }

        /// <summary>PromQL regex matching pods to monitor, e.g. "my-service-.*".</summary>
        public required string PodRegex { get; init; }

        public TimeSpan ScrapeInterval    { get; init; } = TimeSpan.FromSeconds(15);
        public float    AlertThreshold    { get; init; } = 3.0f;
        public float    CriticalThreshold { get; init; } = 6.0f;
        public TimeSpan CooldownDuration  { get; init; } = TimeSpan.FromMinutes(5);

        /// <summary>
        /// Rolling window in snapshots fed to each detector.
        /// Default 21 = ~5 minutes at 15s scrape.
        /// </summary>
        public int ContextSnapshots { get; init; } = 21;

        /// <summary>Called for every scored snapshot (logging, metrics export).</summary>
        public Action<Gpt.AnomalyScore>? OnScore { get; init; }

        /// <summary>Called on non-cancellation errors in the scrape loop.</summary>
        public Action<Exception>? OnError { get; init; }
    }
}
