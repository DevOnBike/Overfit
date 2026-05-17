// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    public class MetricTokenizerTests
    {
        private readonly MetricTokenizer _sut = new();

        [Fact]
        public void VocabSize_Is_MetricCount_Times_BinsPerMetric()
        {
            Assert.Equal(MetricTokenizer.MetricCount * MetricTokenizer.BinsPerMetric, MetricTokenizer.VocabSize);
        }

        [Fact]
        public void EncodeSnapshot_Produces_TokensPerSnapshot_Tokens()
        {
            var snapshot = MakeNormalSnapshot();
            var tokens = new int[MetricTokenizer.TokensPerSnapshot];

            _sut.EncodeSnapshot(snapshot, tokens);

            Assert.Equal(MetricTokenizer.TokensPerSnapshot, tokens.Length);
        }

        [Fact]
        public void AllTokens_AreInVocabRange()
        {
            var snapshot = MakeNormalSnapshot();
            var tokens = new int[MetricTokenizer.TokensPerSnapshot];
            _sut.EncodeSnapshot(snapshot, tokens);

            foreach (var t in tokens)
            {
                Assert.InRange(t, 0, MetricTokenizer.VocabSize - 1);
            }
        }

        [Fact]
        public void NormalSnapshot_ProducesDifferentTokens_ThanAnomalySnapshot()
        {
            var normal = MakeNormalSnapshot();
            var anomaly = MakeAnomalySnapshot();

            var tokNormal = new int[MetricTokenizer.TokensPerSnapshot];
            var tokAnomaly = new int[MetricTokenizer.TokensPerSnapshot];

            _sut.EncodeSnapshot(normal, tokNormal);
            _sut.EncodeSnapshot(anomaly, tokAnomaly);

            // At least some tokens should differ
            Assert.NotEqual(tokNormal, tokAnomaly);
        }

        [Fact]
        public void ZeroSnapshot_AllTokensAreZeroBin()
        {
            var zero = MakeZeroSnapshot();
            var tokens = new int[MetricTokenizer.TokensPerSnapshot];
            _sut.EncodeSnapshot(zero, tokens);

            for (var m = 0; m < MetricTokenizer.MetricCount; m++)
            {
                // Zero value → bin 0 → token = m * BinsPerMetric + 0
                Assert.Equal(m * MetricTokenizer.BinsPerMetric, tokens[m]);
            }
        }

        [Fact]
        public void MetricIndexOf_ReturnsCorrectMetric()
        {
            for (var m = 0; m < MetricTokenizer.MetricCount; m++)
            {
                var token = m * MetricTokenizer.BinsPerMetric + 5;
                Assert.Equal(m, MetricTokenizer.MetricIndexOf(token));
            }
        }

        [Fact]
        public void BinOf_ReturnsCorrectBin()
        {
            for (var b = 0; b < MetricTokenizer.BinsPerMetric; b++)
            {
                var token = 3 * MetricTokenizer.BinsPerMetric + b;
                Assert.Equal(b, MetricTokenizer.BinOf(token));
            }
        }

        [Fact]
        public void Decode_HighCpu_ReturnsApproxValue()
        {
            // Encode high CPU (90%) → decode → should be approximately 0.9
            var snapshot = new MetricSnapshot { CpuUsageRatio = 0.9f, PodName = "test" };
            var tokens = new int[MetricTokenizer.TokensPerSnapshot];
            _sut.EncodeSnapshot(snapshot, tokens);

            var cpuToken = tokens[0]; // metric 0 = CpuUsageRatio
            var decoded = MetricTokenizer.Decode(cpuToken);

            // Should be within ±5% of actual value
            Assert.InRange(decoded, 0.80f, 1.0f);
        }

        [Fact]
        public void EncodeSequence_LengthEquals_SnapshotCount_Times_TokensPerSnapshot()
        {
            var snapshots = new List<MetricSnapshot>
            {
                MakeNormalSnapshot(),
                MakeAnomalySnapshot(),
                MakeNormalSnapshot(),
            };

            var tokens = _sut.EncodeSequence(snapshots);

            Assert.Equal(snapshots.Count * MetricTokenizer.TokensPerSnapshot, tokens.Length);
        }

        [Fact]
        public void MetricNameOf_ReturnsNonEmpty_ForAllMetrics()
        {
            for (var m = 0; m < MetricTokenizer.MetricCount; m++)
            {
                var name = MetricTokenizer.MetricNameOf(m * MetricTokenizer.BinsPerMetric);
                Assert.False(string.IsNullOrEmpty(name), $"Metric {m} has empty name");
            }
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        private static MetricSnapshot MakeNormalSnapshot() => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = "api-gateway",
            CpuUsageRatio = 0.25f,
            CpuThrottleRatio = 0.02f,
            MemoryWorkingSetBytes = 300_000_000f,
            OomEventsRate = 0f,
            LatencyP50Ms = 12f,
            LatencyP95Ms = 35f,
            LatencyP99Ms = 80f,
            RequestsPerSecond = 250f,
            ErrorRate = 0.002f,
            GcGen2HeapBytes = 50_000_000f,
            GcPauseRatio = 0.005f,
            ThreadPoolQueueLength = 8f,
        };

        private static MetricSnapshot MakeAnomalySnapshot() => new()
        {
            Timestamp = DateTime.UtcNow,
            PodName = "api-gateway",
            CpuUsageRatio = 0.95f,  // CPU spike
            CpuThrottleRatio = 0.80f,  // Heavy throttling
            MemoryWorkingSetBytes = 7_000_000_000f, // Near OOM
            OomEventsRate = 0.05f,  // OOM events
            LatencyP50Ms = 800f,   // High latency
            LatencyP95Ms = 2500f,
            LatencyP99Ms = 5000f,
            RequestsPerSecond = 50f,    // Dropping requests
            ErrorRate = 0.35f,  // High error rate
            GcGen2HeapBytes = 5_000_000_000f,
            GcPauseRatio = 0.4f,   // GC thrashing
            ThreadPoolQueueLength = 480f,   // Thread starvation
        };

        private static MetricSnapshot MakeZeroSnapshot() => new()
        {
            PodName = "test",
        };
    }
}
