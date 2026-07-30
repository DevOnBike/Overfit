// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Reading the client's file, and what happens to the parts of it that cannot be read.
    ///
    /// <para>The thing under test is really a policy: <b>an unreadable entry is dropped and reported, never
    /// defaulted.</b> A threshold that quietly became zero is a gate that quietly stopped gating, and this
    /// pipeline has paid for that class of failure repeatedly.</para>
    /// </summary>
    public sealed class AnomalyGuardConfigReaderTests
    {
        [Theory]
        [InlineData("100MB", 100_000_000.0)]
        [InlineData("1GB", 1_000_000_000.0)]
        [InlineData("50ms", 0.05)]
        [InlineData("250us", 0.000_25)]
        [InlineData("2s", 2.0)]
        [InlineData("5%", 0.05)]
        [InlineData("0.01", 0.01)]
        [InlineData(" 100 MB ", 100_000_000.0)]
        public void UnitsAreReadTheWayAHumanWritesThem(string text, double expected)
        {
            Assert.True(MetricQuantity.TryParse(text, out var value));
            Assert.Equal(expected, value, 9);
        }

        /// <summary><c>ms</c> must not be read as <c>m</c> or as <c>s</c>.</summary>
        [Fact]
        public void ALongerUnitWinsOverAShorterOne()
        {
            Assert.True(MetricQuantity.TryParse("50ms", out var ms));
            Assert.True(MetricQuantity.TryParse("50s", out var s));

            Assert.Equal(0.05, ms, 9);
            Assert.Equal(50.0, s, 9);
        }

        [Theory]
        [InlineData("")]
        [InlineData("   ")]
        [InlineData("lots")]
        [InlineData("100 gigabytes")]
        [InlineData("NaN")]
        public void AnUnreadableQuantityIsRefusedRatherThanGuessed(string text)
        {
            Assert.False(MetricQuantity.TryParse(text, out _));
        }

        [Fact]
        public void AKnownFeatureBecomesAQuery()
        {
            var file = new AnomalyGuardConfigFile();
            file.Metrics["LatencyP95Ms"] = new AnomalyGuardConfigFile.MetricEntry
            {
                Source = "http_server_duration_seconds",
                Kind = "HistogramSeconds",
            };

            var map = AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Empty(problems);
            Assert.True(map.IsMapped(MetricIndex.LatencyP95Ms));
            Assert.Contains("histogram_quantile(0.95", map.ToQueryOverrides()[MetricIndex.LatencyP95Ms],
                StringComparison.Ordinal);
        }

        [Fact]
        public void ThresholdsAreKeyedByNameAndCarryUnits()
        {
            var file = new AnomalyGuardConfigFile();
            file.Thresholds["MemoryWorkingSetBytes"] = new AnomalyGuardConfigFile.ThresholdEntry
            {
                MinGap = "100MB",
            };

            var (gap, _) = AnomalyGuardConfigReader.ReadThresholds(file, out var problems);

            Assert.Empty(problems);
            Assert.Equal(100_000_000.0, gap[(int)MetricIndex.MemoryWorkingSetBytes], 3);
            Assert.Equal(0.0, gap[(int)MetricIndex.LatencyP95Ms]);
        }

        /// <summary>The policy: dropped and reported, so the operator finds out at configuration time.</summary>
        [Fact]
        public void AnUnreadableThresholdIsReportedAndTheGateIsOff()
        {
            var file = new AnomalyGuardConfigFile();
            file.Thresholds["MemoryWorkingSetBytes"] = new AnomalyGuardConfigFile.ThresholdEntry
            {
                MinGap = "one hundred megabytes",
            };

            var (gap, _) = AnomalyGuardConfigReader.ReadThresholds(file, out var problems);

            Assert.Single(problems);
            Assert.Contains("gate is OFF", problems[0], StringComparison.Ordinal);
            Assert.Equal(0.0, gap[(int)MetricIndex.MemoryWorkingSetBytes]);
        }

        [Fact]
        public void EveryProblemIsReportedInOnePass()
        {
            var file = new AnomalyGuardConfigFile();
            file.Metrics["NotAFeature"] = new AnomalyGuardConfigFile.MetricEntry { Source = "x" };
            file.Metrics["LatencyP95Ms"] = new AnomalyGuardConfigFile.MetricEntry
            {
                Source = "d",
                Kind = "NotAKind",
            };

            AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Equal(2, problems.Count);
        }

        [Fact]
        public void ACustomMetricIsReadWithItsClassAndSignalKind()
        {
            var file = new AnomalyGuardConfigFile();
            file.CustomMetrics["kafka_consumer_lag"] = new AnomalyGuardConfigFile.CustomEntry
            {
                Source = "kafka_consumergroup_lag",
                Kind = "Gauge",
                LoadSensitive = false,
                Class = "Symptom",
                MinGap = "1000",
                RuleThreshold = "50000",
            };

            var map = AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Empty(problems);
            Assert.Single(map.Custom);

            var binding = map.Custom[0];

            Assert.Equal("kafka_consumer_lag", binding.Name);
            Assert.Equal(PeerSignalKind.LoadIndependent, binding.SignalKind);
            Assert.Equal(SignalClass.Symptom, binding.Class);
            Assert.Equal(1000.0, binding.MinAbsoluteGap, 3);
            Assert.NotNull(binding.Rule);
            Assert.Equal(50_000.0, binding.Rule!.Value.Threshold, 3);

            Assert.Contains("sum by (pod) (kafka_consumergroup_lag",
                map.CustomQueries()["kafka_consumer_lag"], StringComparison.Ordinal);
        }

        /// <summary>
        /// A name that is already a modelled feature belongs under <c>Metrics</c> — putting it under
        /// <c>CustomMetrics</c> would silently exclude it from the learned family.
        /// </summary>
        [Fact]
        public void AModelledNameUnderCustomIsRefused()
        {
            var file = new AnomalyGuardConfigFile();
            file.CustomMetrics["LatencyP95Ms"] = new AnomalyGuardConfigFile.CustomEntry { Source = "x" };

            var map = AnomalyGuardConfigReader.ReadMap(file, out var problems);

            Assert.Empty(map.Custom);
            Assert.Single(problems);
            Assert.Contains("already a modelled feature", problems[0], StringComparison.Ordinal);
        }

        /// <summary>The pre-flight report names both what is missing and what was dropped.</summary>
        [Fact]
        public void DescribeNamesTheBlindSpotsAndTheDroppedEntries()
        {
            var file = new AnomalyGuardConfigFile();
            file.Metrics["RequestsPerSecond"] = new AnomalyGuardConfigFile.MetricEntry
            {
                Source = "http_requests_total",
                Kind = "Counter",
            };
            file.Thresholds["LatencyP95Ms"] = new AnomalyGuardConfigFile.ThresholdEntry { MinGap = "nope" };

            var report = AnomalyGuardConfigReader.Describe(file);

            Assert.Contains("not available in this deployment", report, StringComparison.Ordinal);
            Assert.Contains("DROPPED", report, StringComparison.Ordinal);
        }
    }
}
