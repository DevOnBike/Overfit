// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Does the guard actually <b>use</b> the history, and does using it change a verdict?
    ///
    /// <para><b>This is the test that stops the feature from being shipped and never called.</b> That failure
    /// has already happened once in this subsystem: <c>AnomalyGuard</c> accepted an incident store from early
    /// on while the hosted service constructed it without one, so durable state existed in the library and
    /// nothing deployable reached it. A unit test of <c>MetricHistory</c> would not have caught that, and does
    /// not catch it here either.</para>
    ///
    /// <para>The scenario is the one measured on the live lab: a signal that climbs across the window because
    /// traffic climbs, on every replica together. The cross-peer component cannot subtract it — every replica
    /// rides the curve — so the trend family reports it. A same-hour-yesterday reference can.</para>
    /// </summary>
    public sealed class SeasonalExpectationTests
    {
        private const string Workload = "svc";

        private static readonly DateTimeOffset WindowStart = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void WithoutHistoryTheDailyCurveIsReportedAsATrend()
        {
            var sink = new CapturingSink();

            Guard(sink, history: null).RunCycle(RisingWindow(), WindowStart.AddMinutes(20));

            Assert.Contains(sink.Signals, s => s == nameof(MetricIndex.RequestsPerSecond));
        }

        /// <summary>
        /// The same window, against a workload that has done exactly this at exactly this hour for the last
        /// three days. Nothing about the cluster changed; what changed is that the guard now knows what
        /// normal looks like here.
        /// </summary>
        [Fact]
        public void WithHistoryTheSameClimbIsExpectedAndNotReported()
        {
            var sink = new CapturingSink();

            Guard(sink, history: SeededHistory()).RunCycle(RisingWindow(), WindowStart.AddMinutes(20));

            Assert.True(
                !sink.Signals.Contains(nameof(MetricIndex.RequestsPerSecond)),
                "expected silence, got:\n" + sink.Detail);
        }

        /// <summary>
        /// And the expectation must not deafen it: a climb steeper than the workload's own history is still a
        /// finding. A reference that suppresses everything has not removed the daily curve, it has removed
        /// the detector.
        /// </summary>
        [Fact]
        public void AClimbSteeperThanUsualIsStillReported()
        {
            var sink = new CapturingSink();

            Guard(sink, history: SeededHistory())
                .RunCycle(RisingWindow(scale: 3.0), WindowStart.AddMinutes(20));

            Assert.Contains(sink.Signals, s => s == nameof(MetricIndex.RequestsPerSecond));
        }

        [Fact]
        public void HistoryLearnedInOneRunIsAvailableToTheNext()
        {
            var store = new MemoryStore();

            var first = Guard(new CapturingSink(), history: null, historyStore: store);

            for (var day = 0; day < 3; day++)
            {
                // The WINDOW has to move with the day, not just the observation time: history is stamped from
                // the window's own start, which is what makes a baseline about the hour the data came from
                // rather than about the moment the guard happened to run.
                first.RunCycle(RisingWindow(at: WindowStart.AddDays(day)), WindowStart.AddDays(day).AddMinutes(20));
            }

            // A fresh guard, as after a restart. Nothing is shared but the store — which now carries the
            // baseline and the floor calibration together, since both are "what normal looks like" and
            // neither is much use without the other after a restart.
            var (restored, _, _, _) = LearnedState.Read(store.State);

            Assert.True(restored.TryGet(Workload, MetricIndex.RequestsPerSecond, WindowStart, out var summary));
            Assert.Equal(3, summary.Days);
        }

        /// <summary>
        /// Learning must not be a side effect of an unrelated option.
        ///
        /// <para><c>_history.Observe</c> sat inside the <c>DecomposeCommonMode</c> branch, so turning the
        /// decomposition off - a reasonable thing to do, and something this suite does elsewhere - silently
        /// threw away a week of seasonal learning. Nothing about the option's name suggests it.</para>
        /// </summary>
        [Fact]
        public void HistoryIsLearnedWithTheDecompositionOff()
        {
            var store = new MemoryStore();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = Workload,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                    MinimumHistoryDays = 2,
                    DecomposeCommonMode = false,
                },
                new CapturingSink(),
                IncidentTrackingOptions.Balanced,
                store: null,
                restoredAt: null,
                historyStore: store);

            for (var day = 0; day < 3; day++)
            {
                guard.RunCycle(
                    RisingWindow(at: WindowStart.AddDays(day)), WindowStart.AddDays(day).AddMinutes(20));
            }

            var (restored, _, _, _) = LearnedState.Read(store.State);

            Assert.True(
                restored.TryGet(Workload, MetricIndex.RequestsPerSecond, WindowStart, out var summary),
                "nothing was learned, so DecomposeCommonMode is still switching off an unrelated subsystem");

            Assert.Equal(3, summary.Days);
        }

        private static AnomalyGuard Guard(
            CapturingSink sink, MetricHistory? history, IIncidentStore? historyStore = null)
        {
            var store = historyStore ?? (history is null ? null : new MemoryStore(history.Write()));

            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = Workload,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                    MinimumHistoryDays = 2,
                },
                sink,
                IncidentTrackingOptions.Balanced,
                store: null,
                restoredAt: null,
                historyStore: store);
        }

        /// <summary>
        /// Three previous days on which this workload climbed, at this hour, exactly as it is climbing now.
        ///
        /// <para>Two hours are seeded because the expectation interpolates between them — that is what gives
        /// it a slope, and a slope is what a within-window climb needs cancelling with. The window runs
        /// 12:00→12:20 and rises 100→160, so the hour it sits in must rise at three times that rate: 100 at
        /// 12:00 to 280 at 13:00.</para>
        /// </summary>
        /// <summary>
        /// **`AN-F1`, at the guard rather than at the detector.** A fleet climbing together, against a
        /// history that does NOT predict the climb, must produce no per-pod finding — because the per-pod
        /// trend is judged against the cross-peer component, which removes exactly what every replica shares.
        ///
        /// <para><b>This test exists because the fix had nothing protecting it.</b> Restoring the old
        /// priority (`if (expectation.IsEmpty) { expectation = common; }`) was mutated into `RunTrend` on
        /// 2026-08-10 and **nothing in the suite noticed** — not this class, not the whole `Anomalies`
        /// filter. The other tests here cannot: their history always matches the climb, so the seasonal
        /// reference explains it and the two references agree. The distinguishing case is a history that
        /// does not explain what the fleet is doing.</para>
        ///
        /// <para>Measured at the detector with twelve pods: a +15% shared climb gives 0 findings against the
        /// cross-peer reference and 12 against the seasonal one.</para>
        /// </summary>
        [Fact]
        public void AFleetClimbingTogetherIsNotReportedPerPodEvenWhenHistoryDoesNotExplainIt()
        {
            var sink = new CapturingSink();

            Guard(sink, history: FlatHistory()).RunCycle(RisingWindow(), WindowStart.AddMinutes(20));

            var perPod = sink.Rows.FindAll(r => !r.Contains("(workload)", StringComparison.Ordinal));

            Assert.True(
                perPod.Count == 0,
                "a movement every replica shares was reported against individual pods, which is the AN-F1 "
                + "regression — the per-pod trend is judged against the cross-peer component precisely so "
                + "that it is not:" + Environment.NewLine + sink.Detail);
        }

        /// <summary>
        /// The control, and without it the test above is satisfied by a guard that reports nothing at all:
        /// one replica diverging from a flat fleet must still be named, under the same history.
        /// </summary>
        [Fact]
        public void OnePodDivergingFromAFlatFleetIsStillNamed()
        {
            var sink = new CapturingSink();

            Guard(sink, history: FlatHistory()).RunCycle(OneDivergingPod(), WindowStart.AddMinutes(20));

            Assert.Contains(
                sink.Signals,
                s => s == nameof(MetricIndex.RequestsPerSecond));
        }

        /// <summary>
        /// A history saying "this workload sits at 100 at this hour" — which is what a week of quiet days
        /// produces, and the case where the seasonal reference explains nothing about a fleet that has begun
        /// to move together.
        /// </summary>
        private static MetricHistory FlatHistory()
        {
            var history = new MetricHistory();

            for (var day = 1; day <= 3; day++)
            {
                history.Observe(Workload, MetricIndex.RequestsPerSecond, WindowStart.AddDays(-day), 100.0);
                history.Observe(
                    Workload, MetricIndex.RequestsPerSecond, WindowStart.AddDays(-day).AddHours(1), 100.0);
            }

            return history;
        }

        /// <summary>Three flat replicas and one that climbs — the shape the channel exists to catch.</summary>
        private static MetricWindow OneDivergingPod()
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, WindowStart, TimeSpan.FromSeconds(15));
            var rng = new Random(20260810);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    var climb = pod == 0 ? 100.0 + (60.0 * i / (window.Length - 1.0)) : 100.0;

                    rps[i] = climb * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                }
            }

            return window;
        }

        private static MetricHistory SeededHistory()
        {
            var history = new MetricHistory();

            for (var day = 1; day <= 3; day++)
            {
                history.Observe(Workload, MetricIndex.RequestsPerSecond, WindowStart.AddDays(-day), 100.0);
                history.Observe(
                    Workload, MetricIndex.RequestsPerSecond, WindowStart.AddDays(-day).AddHours(1), 280.0);
            }

            return history;
        }

        /// <summary>
        /// Four replicas whose request rate climbs 60% across the window, together — the shape of a daily
        /// curve, which is real, shared and completely meaningless.
        /// </summary>
        private static MetricWindow RisingWindow(double scale = 1.0, DateTimeOffset? at = null)
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, at ?? WindowStart, TimeSpan.FromSeconds(15));
            var rng = new Random(20260801);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    var climb = 100.0 + (60.0 * scale * i / (window.Length - 1.0));

                    rps[i] = climb * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                }
            }

            return window;
        }

        private sealed class MemoryStore : IIncidentStore
        {
            public MemoryStore(string? state = null)
            {
                State = state;
            }

            public string? State
            {
                get; private set;
            }

            /// <summary>Memory does not fill up in a test; there is nothing to report.</summary>
            public string? LastError => null;

            public string? Load() => State;

            public void Save(string state) => State = state;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<string> Signals { get; } = [];

            /// <summary>
            /// Kept so an assertion can say <b>what</b> fired rather than only that something did. Debugging
            /// this by adjusting the code until the count fell was two wrong guesses; reading the rows
            /// settled it immediately.
            /// </summary>
            public List<string> Rows { get; } = [];

            public string Detail => string.Join("\n", Rows);

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Signals.Add(rows[i].Signal);

                    var who = rows[i].NamesAPod ? rows[i].Pod : "(workload)";

                    Rows.Add($"[{rows[i].Kind}] {who} {rows[i].Signal}: {rows[i].Message}");
                }
            }
        }
    }
}
