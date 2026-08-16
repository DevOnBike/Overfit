// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The memory the guard has never had: what a workload normally does, per signal, per hour, across days.
    /// </summary>
    public sealed class MetricHistoryTests
    {
        private static readonly DateTimeOffset Noon = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void ADayOfObservationsBecomesABaseline()
        {
            var history = new MetricHistory();

            for (var day = 0; day < 5; day++)
            {
                history.Observe("svc", MetricIndex.CpuUsageRatio, Noon.AddDays(day), 0.5 + (0.01 * day));
            }

            Assert.True(history.TryGet("svc", MetricIndex.CpuUsageRatio, Noon.AddDays(9), out var summary));
            Assert.Equal(5, summary.Days);
            Assert.Equal(0.52, summary.Median, 6);
            Assert.Equal(0.54, summary.Newest, 6);
        }

        /// <summary>
        /// At a five-minute cadence one hour produces twelve cycles. Recording all of them would fill a
        /// week's storage with half a day of data and make <c>Days</c> a lie.
        /// </summary>
        [Fact]
        public void RepeatedObservationsInOneHourCountAsOneDay()
        {
            var history = new MetricHistory();

            for (var cycle = 0; cycle < 12; cycle++)
            {
                history.Observe("svc", MetricIndex.CpuUsageRatio, Noon.AddMinutes(5 * cycle), 0.5);
            }

            Assert.True(history.TryGet("svc", MetricIndex.CpuUsageRatio, Noon, out var summary));
            Assert.Equal(1, summary.Days);
        }

        /// <summary>
        /// Hours are separate baselines — that is the entire point. A workload at 03:00 and the same workload
        /// at 13:00 are different normals, and merging them produces an expectation that matches neither.
        /// </summary>
        [Fact]
        public void HoursAreKeptApart()
        {
            var history = new MetricHistory();

            history.Observe("svc", MetricIndex.RequestsPerSecond, Noon, 100.0);
            history.Observe("svc", MetricIndex.RequestsPerSecond, Noon.AddHours(-9), 10.0);

            Assert.True(history.TryGet("svc", MetricIndex.RequestsPerSecond, Noon, out var busy));
            Assert.True(history.TryGet("svc", MetricIndex.RequestsPerSecond, Noon.AddHours(-9), out var quiet));

            Assert.Equal(100.0, busy.Median);
            Assert.Equal(10.0, quiet.Median);
        }

        [Fact]
        public void OnlyTheMostRecentDaysAreKept()
        {
            var history = new MetricHistory();

            for (var day = 0; day < MetricHistory.MaxDays + 5; day++)
            {
                history.Observe("svc", MetricIndex.CpuUsageRatio, Noon.AddDays(day), day);
            }

            Assert.True(history.TryGet("svc", MetricIndex.CpuUsageRatio, Noon, out var summary));
            Assert.Equal(MetricHistory.MaxDays, summary.Days);

            // The oldest days were dropped, not the newest — a store that forgets today is worse than one
            // that forgets last week.
            Assert.Equal(MetricHistory.MaxDays + 4, summary.Newest);
        }

        [Fact]
        public void ItSurvivesARoundTrip()
        {
            var history = new MetricHistory();

            for (var day = 0; day < 4; day++)
            {
                history.Observe("svc", MetricIndex.MemoryWorkingSetBytes, Noon.AddDays(day), 4.0e8 + day);
                history.Observe("svc", MetricIndex.CpuUsageRatio, Noon.AddDays(day).AddHours(3), 0.25);
            }

            var restored = MetricHistory.Read(history.Write());

            Assert.Equal(history.Buckets, restored.Buckets);
            Assert.True(restored.TryGet("svc", MetricIndex.MemoryWorkingSetBytes, Noon, out var memory));
            Assert.Equal(4, memory.Days);
            Assert.Equal(4.0e8 + 3, memory.Newest);
        }

        /// <summary>
        /// A corrupt or absent store must yield an empty history, never an exception. Refusing to start
        /// because a baseline file is unreadable replaces a quiet degradation with an outage.
        /// </summary>
        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("not our format at all")]
        [InlineData("overfit-metric-history\tv1\ngarbage\tlines\there")]
        public void AnUnreadableStoreIsAColdStart(string? state)
        {
            var history = MetricHistory.Read(state);

            Assert.Equal(0, history.Buckets);
        }

        /// <summary>
        /// The expectation must carry the daily curve's <b>slope</b>, not just its level. A flat reference
        /// subtracts the level and leaves the climb, which is precisely the false positive it exists to
        /// remove.
        /// </summary>
        [Fact]
        public void TheExpectationInterpolatesBetweenHoursRatherThanSteppingBetweenThem()
        {
            var history = new MetricHistory();

            for (var day = 1; day <= 3; day++)
            {
                history.Observe("svc", MetricIndex.RequestsPerSecond, Noon.AddDays(-day), 40.0);
                history.Observe("svc", MetricIndex.RequestsPerSecond, Noon.AddDays(-day).AddHours(1), 100.0);
            }

            // Four samples, not five: a fifth would land exactly on 13:00 and therefore need the 14:00
            // bucket to interpolate towards. The window must end strictly inside the last known hour.
            var expectation = new double[4];

            Assert.True(history.TryExpectation(
                "svc", MetricIndex.RequestsPerSecond, Noon, TimeSpan.FromMinutes(15), 2, expectation));

            // 40 at 12:00 rising to 100 at 13:00 — a quarter of the way in is 55, not still 40.
            Assert.Equal(40.0, expectation[0], 6);
            Assert.Equal(55.0, expectation[1], 6);
            Assert.Equal(70.0, expectation[2], 6);
            Assert.Equal(85.0, expectation[3], 6);
        }

        /// <summary>
        /// Extrapolating past the last known hour would invent a continuation of a curve nobody observed.
        /// </summary>
        [Fact]
        public void AWindowRunningPastTheLastKnownHourIsRefused()
        {
            var history = new MetricHistory();

            for (var day = 1; day <= 3; day++)
            {
                history.Observe("svc", MetricIndex.RequestsPerSecond, Noon.AddDays(-day), 40.0);
            }

            Assert.False(history.TryExpectation(
                "svc", MetricIndex.RequestsPerSecond, Noon, TimeSpan.FromMinutes(15), 2, new double[5]));
        }

        /// <summary>
        /// The first observation of an hour stands, because it anchors that hour's interpolation. Keeping the
        /// last would move every anchor to the hour's end and put a phase error into every expectation.
        /// </summary>
        [Fact]
        public void TheFirstObservationOfAnHourIsTheOneKept()
        {
            var history = new MetricHistory();

            history.Observe("svc", MetricIndex.CpuUsageRatio, Noon.AddMinutes(2), 0.10);
            history.Observe("svc", MetricIndex.CpuUsageRatio, Noon.AddMinutes(57), 0.90);

            Assert.True(history.TryGet("svc", MetricIndex.CpuUsageRatio, Noon, out var summary));
            Assert.Equal(0.10, summary.Median, 6);
        }

        /// <summary>
        /// A partial expectation is worse than none — the detector cannot tell which half to trust — so a
        /// refusal must leave the buffer untouched.
        /// </summary>
        [Fact]
        public void AnIncompleteExpectationIsRefusedWithoutWriting()
        {
            var history = new MetricHistory();

            for (var day = 1; day <= 3; day++)
            {
                history.Observe("svc", MetricIndex.RequestsPerSecond, Noon.AddDays(-day), 40.0);
                history.Observe("svc", MetricIndex.RequestsPerSecond, Noon.AddDays(-day).AddHours(1), 80.0);
            }

            var expectation = new double[80];

            // The window runs into the hour after next, for which nothing was ever recorded.
            Assert.False(history.TryExpectation(
                "svc", MetricIndex.RequestsPerSecond, Noon.AddMinutes(30), TimeSpan.FromSeconds(90), 2,
                expectation));

            for (var i = 0; i < expectation.Length; i++)
            {
                Assert.Equal(0.0, expectation[i]);
            }
        }

        [Fact]
        public void OneDayIsNotABaseline()
        {
            var history = new MetricHistory();

            history.Observe("svc", MetricIndex.CpuUsageRatio, Noon, 0.5);

            Assert.False(history.TryExpectation(
                "svc", MetricIndex.CpuUsageRatio, Noon, TimeSpan.FromSeconds(15), 2, new double[4]));
        }

        [Fact]
        public void AWorkloadThatStoppedReportingIsForgotten()
        {
            var history = new MetricHistory();

            history.Observe("gone", MetricIndex.CpuUsageRatio, Noon.AddDays(-30), 0.5);
            history.Observe("live", MetricIndex.CpuUsageRatio, Noon, 0.5);

            Assert.Equal(2, history.Buckets);
            Assert.Equal(1, history.Forget(Noon, TimeSpan.FromDays(14)));
            Assert.Equal(1, history.Buckets);
            Assert.True(history.TryGet("live", MetricIndex.CpuUsageRatio, Noon, out _));
        }

        /// <summary>
        /// Bounded, because a store that grows with every name a cluster has ever produced is a memory leak
        /// inside the monitoring tool.
        /// </summary>
        [Fact]
        public void TheStoreIsBounded()
        {
            var history = new MetricHistory(maxBuckets: 3);

            for (var i = 0; i < 50; i++)
            {
                history.Observe($"svc-{i}", MetricIndex.CpuUsageRatio, Noon, 0.5);
            }

            Assert.Equal(3, history.Buckets);
        }

        [Fact]
        public void NonFiniteObservationsAreIgnored()
        {
            var history = new MetricHistory();

            history.Observe("svc", MetricIndex.CpuUsageRatio, Noon, double.NaN);
            history.Observe("svc", MetricIndex.CpuUsageRatio, Noon, double.PositiveInfinity);

            Assert.Equal(0, history.Buckets);
        }
    }
}
