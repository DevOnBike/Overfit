// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Adaptive;
using DevOnBike.Overfit.Anomalies.Alerting.Abstractions;
using DevOnBike.Overfit.Anomalies.Alerting.Contracts;
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Live;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Maths;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// End-to-end <see cref="LiveMonitoringPipeline"/> validation WITHOUT a live Prometheus:
    /// a scripted <see cref="IRawMetricSource"/> replays benign scrapes (which the cross-pod
    /// base scores as false positives) then an incident, while the pipeline scores through the
    /// adaptive monitor, alerts, and (AutoAdapt) auto-adapts the pod. Asserts the whole chain
    /// ran: benign started elevated → adaptation fired and flattened it → the incident still
    /// produced a Critical alert.
    /// </summary>
    public sealed class LiveMonitoringPipelineTests
    {
        private readonly ITestOutputHelper _out;
        public LiveMonitoringPipelineTests(ITestOutputHelper output) => _out = output;

        private const int ContextSnapshots = 6;
        private const string Pod = "payments-api";

        [LocalOnlyFact]
        public async Task Pipeline_ScrapeScoreAlert_AutoAdaptsAndStillFiresOnIncident()
        {
            var dir = Path.Combine(Path.GetTempPath(), $"overfit_pipeline_{Guid.NewGuid():N}");
            try
            {
                MathUtils.SetSeed(100); using var model = new GPT1Model(new GPT1Config
                {
                    VocabSize = MetricTokenizer.VocabSize,
                    ContextLength = 16 * MetricTokenizer.TokensPerSnapshot,
                    DModel = 32,
                    NHeads = 2,
                    NLayers = 1,
                    DFF = 64,
                    TieWeights = false,
                    PreLayerNorm = true,
                });

                // Scripted scrapes: ~50 benign, then 5 incidents.
                var batches = new List<List<RawMetricSeries>>();
                for (var i = 0; i < 50; i++) { batches.Add(Batch(Normal())); }
                for (var i = 0; i < 5; i++) { batches.Add(Batch(Anomaly())); }

                using var cts = new CancellationTokenSource();
                var source = new ScriptedRawMetricSource(batches, cts);
                var sink = new CapturingSink();
                var scores = new List<float>();

                var options = new LiveMonitoringOptions
                {
                    PrometheusBaseUrl = "unused",
                    PodRegex = "unused",
                    ScrapeInterval = TimeSpan.FromMilliseconds(1),
                    AlertThreshold = 2f,           // alert engine
                    CriticalThreshold = 8f,
                    CooldownDuration = TimeSpan.Zero,
                    AutoAdapt = true,
                    OnScore = s => scores.Add(s.Score),
                    Adaptation = new AdaptivePolicy
                    {
                        AdapterDirectory = dir,
                        ContextSnapshots = ContextSnapshots,
                        AlertThreshold = 2f,       // FP-pressure band
                        CriticalThreshold = 15f,
                        AdaptAfterStreak = 3,
                        MinBenignWindow = 24,
                        BenignWindow = 48,
                        LoRARank = 16,
                        LoRASteps = 300,
                        LoRALearningRate = 1e-2f,
                    },
                };

                await using (var pipeline = LiveMonitoringPipeline.CreateForTest(model, source, options, sink))
                {
                    await pipeline.RunAsync(cts.Token);

                    Assert.True(pipeline.IsAdapted(Pod), "pipeline did not auto-adapt the pod end-to-end.");
                }

                _out.WriteLine(
                    $"scores: first={scores.FirstOrDefault():F2} min={scores.Min():F2} max={scores.Max():F2} n={scores.Count}; " +
                    $"alerts={sink.Events.Count} critical={sink.Events.Count(e => e.Severity == AlertSeverity.Critical)}");

                // Benign started elevated (false-positive pressure)…
                Assert.True(scores.Take(10).Any(s => s >= 2f), "benign never registered as elevated.");
                // …adaptation flattened a later benign read toward zero…
                Assert.True(scores.Any(s => s < 1f), "adaptation did not flatten any benign score.");
                // …and the incident still fired a Critical alert through the pipeline.
                Assert.True(scores.Max() >= 8f, "the incident did not score critically.");
                Assert.Contains(sink.Events, e => e.PodName == Pod && e.Severity == AlertSeverity.Critical);
            }
            finally
            {
                if (Directory.Exists(dir)) { Directory.Delete(dir, recursive: true); }
            }
        }

        // Builds one scrape batch (12 RawMetricSeries, one per metric) from a snapshot.
        private static List<RawMetricSeries> Batch(MetricSnapshot snapshot)
        {
            Span<float> features = stackalloc float[MetricSnapshot.FeatureCount];
            snapshot.WriteFeatureVector(features);
            var ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

            var series = new List<RawMetricSeries>(MetricSnapshot.FeatureCount);
            for (var m = 0; m < MetricSnapshot.FeatureCount; m++)
            {
                var s = new RawMetricSeries { Pod = new PodKey { PodName = snapshot.PodName }, MetricTypeId = (byte)m };
                s.Samples.Add(new RawSample { Timestamp = ts, Value = features[m] });
                series.Add(s);
            }
            return series;
        }

        private static MetricSnapshot Normal() => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = Pod,
            CpuUsageRatio = 0.22f,
            CpuThrottleRatio = 0.02f,
            MemoryWorkingSetBytes = 360_000_000f,
            OomEventsRate = 0f,
            LatencyP50Ms = 13f,
            LatencyP95Ms = 38f,
            LatencyP99Ms = 78f,
            RequestsPerSecond = 270f,
            ErrorRate = 0.003f,
            GcGen2HeapBytes = 52_000_000f,
            GcPauseRatio = 0.004f,
            ThreadPoolQueueLength = 9f,
        };

        private static MetricSnapshot Anomaly() => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = Pod,
            CpuUsageRatio = 0.99f,
            CpuThrottleRatio = 0.65f,
            MemoryWorkingSetBytes = 1_900_000_000f,
            OomEventsRate = 3f,
            LatencyP50Ms = 220f,
            LatencyP95Ms = 1_400f,
            LatencyP99Ms = 2_900f,
            RequestsPerSecond = 12f,
            ErrorRate = 0.42f,
            GcGen2HeapBytes = 1_400_000_000f,
            GcPauseRatio = 0.38f,
            ThreadPoolQueueLength = 240f,
        };

        // Replays scripted scrape batches; cancels the run when exhausted.
        private sealed class ScriptedRawMetricSource : IRawMetricSource
        {
            private readonly Queue<List<RawMetricSeries>> _batches;
            private readonly CancellationTokenSource _cts;

            public ScriptedRawMetricSource(IEnumerable<List<RawMetricSeries>> batches, CancellationTokenSource cts)
            {
                _batches = new Queue<List<RawMetricSeries>>(batches);
                _cts = cts;
            }

            public Task<List<RawMetricSeries>> ReadAsync(CancellationToken ct = default)
            {
                if (_batches.Count == 0)
                {
                    _cts.Cancel();
                    return Task.FromResult(new List<RawMetricSeries>());
                }
                return Task.FromResult(_batches.Dequeue());
            }

            public void Dispose() { }
        }

        private sealed class CapturingSink : IAlertSink
        {
            public List<AlertEvent> Events { get; } = [];

            public Task SendAsync(AlertEvent alert, CancellationToken ct = default)
            {
                Events.Add(alert);
                return Task.CompletedTask;
            }
        }
    }
}
