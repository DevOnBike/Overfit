// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.Anomalies.Adaptive;
using DevOnBike.Overfit.Anomalies.Alerting.Abstractions;
using DevOnBike.Overfit.Anomalies.Alerting.Contracts;
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using GptTrainingConfig = DevOnBike.Overfit.Anomalies.Training.GptTrainingConfig;

namespace DevOnBike.Overfit.Anomalies.Live
{
    /// <summary>
    /// Production live monitoring pipeline with per-pod adaptive learning.
    ///
    /// Flow (every ScrapeInterval):
    ///   Prometheus → MetricSnapshot per pod → AdaptiveAnomalyMonitor.Observe (scores with the
    ///   pod's adapter active) → raw-score alerting (sinks) + adaptation. After each scrape, pods
    ///   showing sustained false-positive pressure are surfaced via OnAdaptationRecommended (or
    ///   auto-adapted when AutoAdapt is set). Per-pod adapters persist as &lt;pod&gt;.lora.bin and
    ///   reload on pod restart, so adaptation survives process/pod churn.
    ///
    /// Alerting is on the RAW GPT surprise score (nats/token) against
    /// <see cref="LiveMonitoringOptions.AlertThreshold"/> / <see cref="LiveMonitoringOptions.CriticalThreshold"/>,
    /// with a per-pod cooldown — the legacy [0,1]-normalised <c>AlertEngine</c> (autoencoder era)
    /// is not used here, since the GPT score is unbounded nats, not a normalised reconstruction error.
    ///
    /// Usage:
    ///   await using var pipeline = LiveMonitoringPipeline.Create(
    ///       options: new LiveMonitoringOptions { …, Adaptation = new AdaptivePolicy { AdapterDirectory = "/var/overfit/adapters" } },
    ///       trainingConfig: GptTrainingConfig.Production,
    ///       checkpointPath: "k8s_anomaly_production.bin",
    ///       new TeamsAlertSink(webhookUrl));
    ///   await pipeline.RunAsync(cancellationToken);
    /// </summary>
    public sealed class LiveMonitoringPipeline : IAsyncDisposable
    {
        private readonly IRawMetricSource _source;
        private readonly GPT1Model _model;
        private readonly AdaptiveAnomalyMonitor _monitor;
        private readonly LiveMonitoringOptions _options;
        private readonly IAlertSink[] _sinks;
        private readonly HashSet<string> _recommended = new(StringComparer.Ordinal);
        private readonly Dictionary<string, DateTime> _lastAlert = new(StringComparer.Ordinal);
        private bool _disposed;

        private LiveMonitoringPipeline(
            IRawMetricSource source,
            GPT1Model model,
            AdaptiveAnomalyMonitor monitor,
            LiveMonitoringOptions options,
            IAlertSink[] sinks)
        {
            _source = source;
            _model = model;
            _monitor = monitor;
            _options = options;
            _sinks = sinks;
        }

        /// <summary>Creates a live monitoring pipeline from a trained checkpoint.</summary>
        public static LiveMonitoringPipeline Create(
            LiveMonitoringOptions options,
            GptTrainingConfig trainingConfig,
            string checkpointPath,
            params IAlertSink[] sinks)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(options.Adaptation);
            ArgumentNullException.ThrowIfNull(trainingConfig);
            ArgumentNullException.ThrowIfNull(checkpointPath);

            if (!File.Exists(checkpointPath))
            {
                throw new FileNotFoundException($"GPT checkpoint not found: {checkpointPath}");
            }

            var gptConfig = new GPT1Config
            {
                VocabSize = MetricTokenizer.VocabSize,
                ContextLength = trainingConfig.ContextLength,
                DModel = trainingConfig.DModel,
                NHeads = trainingConfig.NHeads,
                NLayers = trainingConfig.NLayers,
                DFF = trainingConfig.DModel * 4,
                TieWeights = false,
                PreLayerNorm = true,
            };

            var model = new GPT1Model(gptConfig);
            model.Eval();
            using (var fs = File.OpenRead(checkpointPath))
            using (var br = new BinaryReader(fs))
            {
                model.Load(br);
            }

            var monitor = new AdaptiveAnomalyMonitor(model, options.Adaptation);

            var sourceConfig = new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = options.PrometheusBaseUrl,
                PodRegex = options.PodRegex,
                ScrapeInterval = options.ScrapeInterval,
            };
            var source = new PrometheusMetricSource(sourceConfig);

            return new LiveMonitoringPipeline(source, model, monitor, options, sinks);
        }

        /// <summary>
        /// Test seam: builds the pipeline over an already-loaded model and an injected
        /// <see cref="IRawMetricSource"/> (e.g. a scripted/in-memory source), so the full
        /// scrape → score → alert → adaptive lifecycle can run without a live Prometheus
        /// instance or a checkpoint file. The pipeline takes ownership of <paramref name="model"/>
        /// and <paramref name="source"/> (disposed in <see cref="DisposeAsync"/>).
        /// </summary>
        internal static LiveMonitoringPipeline CreateForTest(
            GPT1Model model,
            IRawMetricSource source,
            LiveMonitoringOptions options,
            params IAlertSink[] sinks)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(source);
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(options.Adaptation);

            var monitor = new AdaptiveAnomalyMonitor(model, options.Adaptation);
            return new LiveMonitoringPipeline(source, model, monitor, options, sinks);
        }

        /// <summary>Pods the monitor currently recommends adapting (sustained false-positive pressure).</summary>
        public IReadOnlyList<string> PodsNeedingAdaptation() => _monitor.PodsNeedingAdaptation();

        /// <summary>Adapts a pod on its buffered benign window (operator action). Returns the adapter path.</summary>
        public string Adapt(string podName) => _monitor.Adapt(podName);

        /// <summary>True if the pod currently has a per-pod adapter loaded.</summary>
        public bool IsAdapted(string podName) => _monitor.IsAdapted(podName);

        /// <summary>
        /// Runs the monitoring loop until cancellation. Scrapes every ScrapeInterval, scores each
        /// pod through the adaptive monitor, raises alerts on the raw score, and surfaces/auto-applies
        /// per-pod adaptation.
        /// </summary>
        public async Task RunAsync(CancellationToken ct = default)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var scrapeTime = DateTime.UtcNow;
                    var series = await _source.ReadAsync(ct).ConfigureAwait(false);
                    var snapshots = ConvertToSnapshots(series, scrapeTime);

                    foreach (var snapshot in snapshots)
                    {
                        var result = _monitor.Observe(snapshot);
                        if (result.IsWarmup)
                        {
                            continue;
                        }

                        _options.OnScore?.Invoke(result);
                        await MaybeAlertAsync(result, scrapeTime, ct).ConfigureAwait(false);
                    }

                    HandleAdaptation();

                    await Task.Delay(_options.ScrapeInterval, ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _options.OnError?.Invoke(ex);
                }
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            _monitor.Dispose();   // un-merges adapters + disposes detectors, leaving the model clean
            _source.Dispose();
            _model.Dispose();
            await ValueTask.CompletedTask.ConfigureAwait(false);
        }

        // ── Private ──────────────────────────────────────────────────────────

        // Raw-score alerting with a per-pod cooldown — dispatches to every sink.
        private async Task MaybeAlertAsync(AnomalyScore result, DateTime now, CancellationToken ct)
        {
            if (_sinks.Length == 0 || result.Score < _options.AlertThreshold)
            {
                return;
            }
            if (_lastAlert.TryGetValue(result.PodName, out var last) && now - last < _options.CooldownDuration)
            {
                return;
            }
            _lastAlert[result.PodName] = now;

            var evt = new AlertEvent
            {
                PodName = result.PodName,
                AnomalyScore = result.Score,
                ReconstructionMse = result.Score,
                DetectedAt = now,
                Severity = result.Score >= _options.CriticalThreshold ? AlertSeverity.Critical : AlertSeverity.Warning,
            };
            foreach (var sink in _sinks)
            {
                await sink.SendAsync(evt, ct).ConfigureAwait(false);
            }
        }

        // Auto-adapt recommended pods (unattended), else notify the operator once per pod.
        private void HandleAdaptation()
        {
            foreach (var pod in _monitor.PodsNeedingAdaptation())
            {
                if (_options.AutoAdapt)
                {
                    try
                    {
                        _monitor.Adapt(pod);
                    }
                    catch (Exception ex)
                    {
                        _options.OnError?.Invoke(ex);
                    }
                }
                else if (_recommended.Add(pod))
                {
                    _options.OnAdaptationRecommended?.Invoke(pod);
                }
            }
        }

        private static List<MetricSnapshot> ConvertToSnapshots(
            IReadOnlyList<RawMetricSeries> series, DateTime timestamp)
        {
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
                    Timestamp = timestamp,
                    PodName = podName,
                    CpuUsageRatio = f[0],
                    CpuThrottleRatio = f[1],
                    MemoryWorkingSetBytes = f[2],
                    OomEventsRate = f[3],
                    LatencyP50Ms = f[4],
                    LatencyP95Ms = f[5],
                    LatencyP99Ms = f[6],
                    RequestsPerSecond = f[7],
                    ErrorRate = f[8],
                    GcGen2HeapBytes = f[9],
                    GcPauseRatio = f[10],
                    ThreadPoolQueueLength = f[11],
                });
            }
            return result;
        }

        private static int MetricTypeIdToFeatureIndex(byte id) => id switch
        {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 7,
            8 => 8,
            9 => 9,
            10 => 10,
            11 => 11,
            _ => -1
        };
    }
}
