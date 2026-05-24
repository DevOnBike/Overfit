// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Adaptive;
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Exercises the per-pod adaptive lifecycle (<see cref="AdaptiveAnomalyMonitor"/>): a base
    /// scores a pod's benign regime as elevated (false-positive pressure) → the monitor
    /// recommends adaptation → operator adapts on the buffered benign window → the benign
    /// score flattens below the alert band while an injected incident still fires; plus
    /// per-pod isolation (adapting pod A leaves pod B on the base). Thresholds are widened
    /// ([2, 15)) so the lifecycle is robust to the tiny random base's exact scores.
    /// </summary>
    public sealed class AdaptiveAnomalyMonitorTests
    {
        private readonly ITestOutputHelper _out;
        public AdaptiveAnomalyMonitorTests(ITestOutputHelper output) => _out = output;

        private const int ContextSnapshots = 6;

        private static GPT1Model TinyModel() => new(new GPT1Config
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

        private static AdaptivePolicy Policy(string dir) => new()
        {
            AdapterDirectory = dir,
            ContextSnapshots = ContextSnapshots,
            AlertThreshold = 2f,
            CriticalThreshold = 15f,
            AdaptAfterStreak = 3,
            MinBenignWindow = 24,
            BenignWindow = 48,
            LoRARank = 16,
            LoRASteps = 300,
            LoRALearningRate = 1e-2f,
        };

        [Fact]
        public void Lifecycle_RecommendsAdaptation_ThenFlattensBenign_StillFiresOnIncident()
        {
            var dir = Path.Combine(Path.GetTempPath(), $"overfit_adaptive_{Guid.NewGuid():N}");
            try
            {
                using var model = TinyModel();
                using var monitor = new AdaptiveAnomalyMonitor(model, Policy(dir));

                // Feed a steady benign regime — the un-adapted base scores it elevated.
                var baseBenign = 0f;
                for (var i = 0; i < ContextSnapshots * 2 + 40; i++)
                {
                    var r = monitor.Observe(Normal("payments-api"));
                    if (!r.IsWarmup) { baseBenign = r.Score; }
                }
                _out.WriteLine($"base benign={baseBenign:F3}, needsAdapt={monitor.NeedsAdaptation("payments-api")}");

                Assert.True(baseBenign >= 2f, $"base did not show false-positive pressure: {baseBenign:F3}.");
                Assert.True(monitor.NeedsAdaptation("payments-api"), "monitor did not recommend adaptation.");

                // Operator adapts on the buffered benign window.
                monitor.Adapt("payments-api");
                Assert.True(monitor.IsAdapted("payments-api"));
                Assert.False(monitor.NeedsAdaptation("payments-api"));

                // Re-stream benign → flattened below the alert band.
                var adaptedBenign = 0f;
                for (var i = 0; i < ContextSnapshots * 2; i++)
                {
                    var r = monitor.Observe(Normal("payments-api"));
                    if (!r.IsWarmup) { adaptedBenign = r.Score; }
                }
                _out.WriteLine($"adapted benign={adaptedBenign:F3}");

                Assert.True(adaptedBenign < baseBenign, $"adaptation did not lower benign: {baseBenign:F3} → {adaptedBenign:F3}.");
                Assert.True(adaptedBenign < 2f, $"adapted benign still in alert band: {adaptedBenign:F3}.");

                // A real incident still fires far above the adapted benign.
                var incident = monitor.Observe(Anomaly("payments-api"));
                _out.WriteLine($"incident={incident.Score:F3} worst={incident.WorstMetric}");
                Assert.False(incident.IsWarmup);
                Assert.True(incident.Score > MathF.Max(adaptedBenign * 3f, 5f),
                    $"adaptation blinded the detector: adapted-benign={adaptedBenign:F3}, incident={incident.Score:F3}.");
            }
            finally
            {
                if (Directory.Exists(dir)) { Directory.Delete(dir, recursive: true); }
            }
        }

        [Fact]
        public void Adapt_OnePod_DoesNotChangeAnotherPod()
        {
            var dir = Path.Combine(Path.GetTempPath(), $"overfit_adaptive_{Guid.NewGuid():N}");
            try
            {
                using var model = TinyModel();
                using var monitor = new AdaptiveAnomalyMonitor(model, Policy(dir));

                // Warm + buffer benign for BOTH pods; capture pod-b's base benign.
                float aBenign = 0f, bBenign = 0f;
                for (var i = 0; i < ContextSnapshots * 2 + 30; i++)
                {
                    var ra = monitor.Observe(Normal("pod-a"));
                    var rb = monitor.Observe(Normal("pod-b"));
                    if (!ra.IsWarmup) { aBenign = ra.Score; }
                    if (!rb.IsWarmup) { bBenign = rb.Score; }
                }

                monitor.Adapt("pod-a");
                Assert.True(monitor.IsAdapted("pod-a"));
                Assert.False(monitor.IsAdapted("pod-b"));

                // pod-a flattens; pod-b (no adapter) stays on the base — the adapter swap
                // must not leak pod-a's delta into pod-b's scoring.
                var aAfter = monitor.Observe(Normal("pod-a")).Score;
                var bAfter = monitor.Observe(Normal("pod-b")).Score;
                _out.WriteLine($"pod-a {aBenign:F3}→{aAfter:F3}, pod-b {bBenign:F3}→{bAfter:F3}");

                Assert.True(aAfter < aBenign, "pod-a did not flatten after its own adaptation.");
                Assert.True(MathF.Abs(bAfter - bBenign) < 0.5f,
                    $"pod-a's adapter leaked into pod-b: {bBenign:F3} → {bAfter:F3}.");
            }
            finally
            {
                if (Directory.Exists(dir)) { Directory.Delete(dir, recursive: true); }
            }
        }

        [Fact]
        public void Adapter_PersistsAndReloads_OnPodRestart()
        {
            var dir = Path.Combine(Path.GetTempPath(), $"overfit_adaptive_{Guid.NewGuid():N}");
            try
            {
                using var model = TinyModel();

                // First monitor: adapt pod-a, then dispose (un-merges, leaving the base clean;
                // the per-pod adapter .bin stays on disk).
                using (var monitor1 = new AdaptiveAnomalyMonitor(model, Policy(dir)))
                {
                    for (var i = 0; i < ContextSnapshots * 2 + 30; i++) { monitor1.Observe(Normal("pod-a")); }
                    monitor1.Adapt("pod-a");
                    Assert.True(monitor1.IsAdapted("pod-a"));
                }

                // Restart: a new monitor over the SAME base model + adapter directory must
                // auto-reload pod-a's adapter from disk (and NOT apply it to other pods).
                using var monitor2 = new AdaptiveAnomalyMonitor(model, Policy(dir));

                var reloaded = 0f;
                for (var i = 0; i < ContextSnapshots * 2; i++)
                {
                    var r = monitor2.Observe(Normal("pod-a"));
                    if (!r.IsWarmup) { reloaded = r.Score; }
                }
                var fresh = 0f;
                for (var i = 0; i < ContextSnapshots * 2; i++)
                {
                    var r = monitor2.Observe(Normal("pod-never-seen"));
                    if (!r.IsWarmup) { fresh = r.Score; }
                }
                _out.WriteLine($"reloaded pod-a={reloaded:F3}, fresh pod={fresh:F3}");

                Assert.True(monitor2.IsAdapted("pod-a"), "adapter did not reload on restart.");
                Assert.False(monitor2.IsAdapted("pod-never-seen"));
                Assert.True(reloaded < fresh,
                    $"reloaded adapter did not flatten pod-a vs the base: reloaded={reloaded:F3}, fresh={fresh:F3}.");
                Assert.True(reloaded < 2f, $"reloaded pod-a still elevated: {reloaded:F3}.");
            }
            finally
            {
                if (Directory.Exists(dir)) { Directory.Delete(dir, recursive: true); }
            }
        }

        private static MetricSnapshot Normal(string pod) => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = pod,
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

        private static MetricSnapshot Anomaly(string pod) => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = pod,
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
    }
}
