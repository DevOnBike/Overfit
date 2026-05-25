// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Baseline;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Fast, deterministic unit tests for the <see cref="EwmaAnomalyDetector"/> classical
    /// baseline: warmup gating, low scores on a noisy-but-normal regime, and high scores
    /// with correct worst-metric attribution on injected OOM / latency anomalies.
    /// </summary>
    public sealed class EwmaAnomalyDetectorTests
    {
        private const int Warmup = 12;

        [Fact]
        public void Warmup_GatesScoring_UntilSeeded()
        {
            var d = new EwmaAnomalyDetector(warmupSnapshots: Warmup);
            var rng = new Random(1);

            for (var i = 0; i < Warmup; i++)
            {
                Assert.True(d.Score(NoisyNormal(rng)).IsWarmup, $"snapshot {i} should be warmup.");
            }
            Assert.True(d.WindowFilled);
            Assert.False(d.Score(NoisyNormal(rng)).IsWarmup);
        }

        [Fact]
        public void NoisyNormalRegime_ScoresLow()
        {
            var d = new EwmaAnomalyDetector(warmupSnapshots: Warmup);
            var rng = new Random(7);

            var maxPostWarmup = 0f;
            var sum = 0f;
            var n = 0;
            for (var i = 0; i < 80; i++)
            {
                var r = d.Score(NoisyNormal(rng));
                if (!r.IsWarmup) { maxPostWarmup = MathF.Max(maxPostWarmup, r.Score); sum += r.Score; n++; }
            }

            // Small Gaussian jitter → per-metric z ≈ 1 → mean ½z² is low on average. EWMA is
            // twitchy (σ lags transient jitter), so an occasional sample reaches ~2; assert
            // the regime is low on average and never enters the strong-anomaly band.
            Assert.True(sum / n < 1.0f, $"normal regime mean too high: {sum / n:F3}.");
            Assert.True(maxPostWarmup < 3.5f, $"normal regime peaked into the anomaly band: max={maxPostWarmup:F3}.");
        }

        [Fact]
        public void InjectedOom_ScoresHigh_AndAttributesToOom()
        {
            var d = new EwmaAnomalyDetector(warmupSnapshots: Warmup);
            var rng = new Random(3);

            var normal = 0f;
            for (var i = 0; i < 40; i++)
            {
                var r = d.Score(NoisyNormal(rng));
                if (!r.IsWarmup) { normal = r.Score; }
            }

            var oom = NormalBaseline() with { OomEventsRate = 4f };  // normally a hard 0 — maximal surprise
            var result = d.Score(oom);

            Assert.False(result.IsWarmup);
            Assert.True(result.Score > 5f * MathF.Max(normal, 1e-3f),
                $"OOM did not separate from normal: normal={normal:F3}, oom={result.Score:F3}.");
            Assert.Equal("oom_events_rate", result.WorstMetric);
        }

        [Fact]
        public void InjectedLatencySpike_AttributesToLatency()
        {
            var d = new EwmaAnomalyDetector(warmupSnapshots: Warmup);
            var rng = new Random(5);

            for (var i = 0; i < 40; i++) { d.Score(NoisyNormal(rng)); }

            var spike = NormalBaseline() with { LatencyP99Ms = 3_000f };  // ~38× the normal p99
            var result = d.Score(spike);

            Assert.False(result.IsWarmup);
            Assert.Contains("latency", result.WorstMetric, StringComparison.Ordinal);
        }

        // ── Helpers ─────────────────────────────────────────────────────────
        private static MetricSnapshot NormalBaseline() => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = "payments-api",
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

        // Small multiplicative Gaussian jitter so the EWMA learns a real (non-zero) σ
        // on every metric except oom_events_rate (held at a hard 0, as in production).
        private static MetricSnapshot NoisyNormal(Random rng)
        {
            float J(float v) => v * (1f + 0.05f * (float)NextGaussian(rng));
            var b = NormalBaseline();
            return b with
            {
                CpuUsageRatio = J(b.CpuUsageRatio),
                CpuThrottleRatio = J(b.CpuThrottleRatio),
                MemoryWorkingSetBytes = J(b.MemoryWorkingSetBytes),
                LatencyP50Ms = J(b.LatencyP50Ms),
                LatencyP95Ms = J(b.LatencyP95Ms),
                LatencyP99Ms = J(b.LatencyP99Ms),
                RequestsPerSecond = J(b.RequestsPerSecond),
                ErrorRate = J(b.ErrorRate),
                GcGen2HeapBytes = J(b.GcGen2HeapBytes),
                GcPauseRatio = J(b.GcPauseRatio),
                ThreadPoolQueueLength = J(b.ThreadPoolQueueLength),
            };
        }

        private static double NextGaussian(Random rng)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
    }
}
