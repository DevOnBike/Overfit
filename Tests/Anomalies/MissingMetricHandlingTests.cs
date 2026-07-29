// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Baseline;
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// A metric that returned no series is now carried as <see cref="float.NaN"/> rather than zero, so that
    /// "the query matched nothing" stops being indistinguishable from "the value is zero". These pin the
    /// consequences of that at every point where a feature is read.
    /// </summary>
    public sealed class MissingMetricHandlingTests
    {
        [Fact]
        public void OneMissingFeature_DoesNotPoisonTheEwmaBaselineForever()
        {
            // The failure this prevents: NaN seeded into the mean makes every later arithmetic operation NaN,
            // and nothing downstream recovers. The detector would report NaN scores for the rest of its life.
            var detector = new EwmaAnomalyDetector(warmupSnapshots: 5);

            for (var i = 0; i < 40; i++)
            {
                var score = detector.Score(Snapshot(cpu: 0.4f, latencyP95: 120f, throttle: float.NaN));

                Assert.False(float.IsNaN(score.Score), $"score went NaN on sample {i}");
            }
        }

        [Fact]
        public void AMissingFeature_IsNotScoredAsADeviationFromZero()
        {
            // Scoring an absent feature against a zero mean would invent a deviation out of an absence.
            var withGap = new EwmaAnomalyDetector(warmupSnapshots: 5);
            var complete = new EwmaAnomalyDetector(warmupSnapshots: 5);

            AnomalyScore gapScore = default;
            AnomalyScore fullScore = default;

            for (var i = 0; i < 30; i++)
            {
                gapScore = withGap.Score(Snapshot(cpu: 0.4f, latencyP95: 120f, throttle: float.NaN));
                fullScore = complete.Score(Snapshot(cpu: 0.4f, latencyP95: 120f, throttle: 0.02f));
            }

            Assert.False(gapScore.IsWarmup);
            Assert.True(gapScore.Score < 0.5f, $"steady input with one gap scored {gapScore.Score}");
            Assert.True(fullScore.Score < 0.5f, $"steady complete input scored {fullScore.Score}");
        }

        [Fact]
        public void AFeatureThatStartsReportingLate_IsSeededByItsFirstFiniteValue()
        {
            // The metric is absent for the first ten scrapes and then appears — which is exactly what a pod
            // acquiring a CPU limit, or an exporter coming up late, looks like.
            var detector = new EwmaAnomalyDetector(warmupSnapshots: 3);

            for (var i = 0; i < 10; i++)
            {
                detector.Score(Snapshot(cpu: 0.4f, latencyP95: 120f, throttle: float.NaN));
            }

            for (var i = 0; i < 20; i++)
            {
                var score = detector.Score(Snapshot(cpu: 0.4f, latencyP95: 120f, throttle: 0.05f));

                Assert.False(float.IsNaN(score.Score));
                Assert.True(score.Score < 1.0f, $"late-arriving feature scored {score.Score} on sample {i}");
            }
        }

        [Fact]
        public void EveryFeatureMissing_ReportsWarmupRatherThanHealthy()
        {
            // Reporting 0 would say "healthy" about a snapshot that could not be read at all.
            var detector = new EwmaAnomalyDetector(warmupSnapshots: 2);

            for (var i = 0; i < 10; i++)
            {
                var score = detector.Score(Blank());

                Assert.True(score.IsWarmup, $"a fully unreadable snapshot was scored as a verdict on sample {i}");
            }
        }

        [Fact]
        public void TheTokenizer_EncodesAMissingFeatureDeterministically()
        {
            // NaN must not fall through to (int)NaN by accident. It lands in the lowest bin by decision —
            // the vocabulary has no "missing" symbol — and that has to be stable, not incidental.
            var tokens = new int[MetricTokenizer.TokensPerSnapshot];

            new MetricTokenizer().EncodeSnapshot(Snapshot(cpu: 0.4f, latencyP95: 120f, throttle: float.NaN), tokens, 0);

            var throttleToken = tokens[(int)MetricIndex.CpuThrottleRatio];

            Assert.Equal((int)MetricIndex.CpuThrottleRatio * MetricTokenizer.BinsPerMetric, throttleToken);
            Assert.Equal((int)MetricIndex.CpuThrottleRatio, MetricTokenizer.MetricIndexOf(throttleToken));
        }

        [Fact]
        public void TheSource_ReportsWhichMetricsHaveNoQueryAtAll()
        {
            // The mechanism that keeps a missing feature from being trained on unnoticed.
            var config = new PrometheusMetricSourceConfig
            {
                PrometheusBaseUrl = "http://127.0.0.1:9090",
                PodRegex = "overfit-server-.*",
                Namespace = "overfit",
                DataCenterLabel = string.Empty,
                QueryOverrides = new Dictionary<MetricIndex, string>
                {
                    [MetricIndex.ErrorRate] = string.Empty,
                    [MetricIndex.GcPauseRatio] = "   "
                }
            };

            using var source = new PrometheusMetricSource(config);

            Assert.False(source.IsMapped(MetricIndex.ErrorRate));
            Assert.False(source.IsMapped(MetricIndex.GcPauseRatio));
            Assert.True(source.IsMapped(MetricIndex.CpuUsageRatio));
            Assert.Equal(0, source.SeriesReturned(MetricIndex.CpuUsageRatio));
        }

        private static MetricSnapshot Blank()
        {
            return new MetricSnapshot
            {
                Timestamp = new DateTime(2026, 7, 28, 12, 0, 0, DateTimeKind.Utc),
                PodName = "pod-a",
                CpuUsageRatio = float.NaN,
                CpuThrottleRatio = float.NaN,
                MemoryWorkingSetBytes = float.NaN,
                OomEventsRate = float.NaN,
                LatencyP50Ms = float.NaN,
                LatencyP95Ms = float.NaN,
                LatencyP99Ms = float.NaN,
                RequestsPerSecond = float.NaN,
                ErrorRate = float.NaN,
                GcGen2HeapBytes = float.NaN,
                GcPauseRatio = float.NaN,
                ThreadPoolQueueLength = float.NaN
            };
        }

        private static MetricSnapshot Snapshot(float cpu, float latencyP95, float throttle)
        {
            return new MetricSnapshot
            {
                Timestamp = new DateTime(2026, 7, 28, 12, 0, 0, DateTimeKind.Utc),
                PodName = "pod-a",
                CpuUsageRatio = cpu,
                CpuThrottleRatio = throttle,
                MemoryWorkingSetBytes = 4.2e8f,
                OomEventsRate = 0f,
                LatencyP50Ms = 40f,
                LatencyP95Ms = latencyP95,
                LatencyP99Ms = 300f,
                RequestsPerSecond = 12f,
                ErrorRate = 0f,
                GcGen2HeapBytes = 2.1e8f,
                GcPauseRatio = 0.01f,
                ThreadPoolQueueLength = 2f
            };
        }
    }
}
