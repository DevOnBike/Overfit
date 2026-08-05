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
    /// The floor that applies when nobody configured one.
    ///
    /// <para><b>Day one is the case this exists for.</b> An absent floor means the gate is off, and a guard
    /// with every absolute gate off is the configuration measured at 209 false incidents a day on a lab where
    /// nothing was wrong. Until now the calibrated figure was only ever printed for a human to transcribe,
    /// which means it protected nothing until somebody got round to it.</para>
    /// </summary>
    public sealed class CalibratedFloorFallbackTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// Two arms over the <b>same</b> windows, because a before-and-after inside one run cannot work here:
        /// the calibrator has a usable sample by the third cycle, so the floor is already in force during any
        /// "first half" a single run could be split into. The first version of this test compared halves and
        /// came back 0 against 0 — vacuous, and it would have passed just as happily if the feature did
        /// nothing at all.
        /// </summary>
        [Fact]
        public void TheLearnedFloorSilencesNoiseThatNoConfiguredFloorWouldHave()
        {
            var withFloor = Run(apply: true);
            var without = Run(apply: false);

            Assert.True(without > 0,
                "the population produced no incidents even with every absolute gate off, so this proves nothing");

            Assert.True(withFloor < without,
                $"expected the learned floor to quieten the run: {withFloor} incidents with it, {without} without");
        }

        private static int Run(bool apply)
        {
            var sink = new CapturingSink();
            var guard = Guard(sink, apply);

            for (var cycle = 0; cycle < 40; cycle++)
            {
                sink.Cycle = cycle;
                guard.RunCycle(Healthy(cycle), T0.AddMinutes(5 * cycle));
            }

            return sink.OpenedFrom(0);
        }

        /// <summary>
        /// An explicit floor wins even when it is lower than the calibrated one. A configured value is a
        /// decision somebody made; overriding it silently would make the configuration a suggestion.
        /// </summary>
        [Fact]
        public void AnExplicitFloorIsNotOverriddenByAHigherCalibratedOne()
        {
            var floors = new double[(int)MetricIndex.Count];
            floors[(int)MetricIndex.GcGen2HeapBytes] = 1.0;

            var sink = new CapturingSink();
            var guard = Guard(sink, apply: true, gapFloors: floors);

            for (var cycle = 0; cycle < 40; cycle++)
            {
                sink.Cycle = cycle;
                guard.RunCycle(Healthy(cycle), T0.AddMinutes(5 * cycle));
            }

            // A floor of one byte gates nothing, so the noise survives — which is the point: the operator's
            // number was honoured rather than quietly replaced.
            Assert.True(sink.OpenedFrom(20) > 0);
        }

        /// <summary>
        /// What is learned has to survive the guard's own restart, or a rollout of the monitoring tool
        /// produces a burst of noise from the monitoring tool.
        /// </summary>
        [Fact]
        public void CalibrationSurvivesARestart()
        {
            var store = new MemoryStore();
            var first = Guard(new CapturingSink(), apply: true, historyStore: store);

            for (var cycle = 0; cycle < 40; cycle++)
            {
                first.RunCycle(Healthy(cycle), T0.AddMinutes(5 * cycle));
            }

            var beforeRestart = first.FloorProposals[(int)MetricIndex.GcGen2HeapBytes];

            Assert.True(beforeRestart.IsUsable);

            var restarted = Guard(new CapturingSink(), apply: true, historyStore: store);
            var afterRestart = restarted.FloorProposals[(int)MetricIndex.GcGen2HeapBytes];

            Assert.True(afterRestart.IsUsable, "the calibration did not survive the restart");
            Assert.Equal(beforeRestart.ProposedMinAbsoluteGap, afterRestart.ProposedMinAbsoluteGap, 6);
        }

        private static AnomalyGuard Guard(
            CapturingSink sink, bool apply, IReadOnlyList<double>? gapFloors = null,
            IIncidentStore? historyStore = null)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    ApplyCalibratedFloors = apply,
                    MinAbsoluteGap = gapFloors,
                    MinAbsoluteTrendChange = gapFloors,

                    // Off: history would supply a second, independent reason for the run to quieten and the
                    // measurement here is about the floor.
                    MinimumHistoryDays = 0,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced,
                store: null,
                restoredAt: null,
                historyStore: historyStore);
        }

        /// <summary>
        /// Twelve replicas at the lab's own scale — a gen2 heap of a few megabytes — where every replica
        /// carries a persistent offset. Nothing is wrong; the offsets are what allocation history looks like.
        /// </summary>
        private static MetricWindow Healthy(int cycle)
        {
            var names = new List<string>(12);

            for (var p = 0; p < 12; p++)
            {
                names.Add($"pod-{p}");
            }

            var window = new MetricWindow(names, 80, T0.AddMinutes(5 * cycle), TimeSpan.FromSeconds(15));
            var rng = new Random(20260801 + cycle);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var heap = window.Series(pod, MetricIndex.GcGen2HeapBytes);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);
                var level = 4.0e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.5));

                for (var i = 0; i < window.Length; i++)
                {
                    heap[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.04));
                    rps[i] = 6.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.05));
                }
            }

            return window;
        }

        private sealed class MemoryStore : IIncidentStore
        {
            private string? _state;

            /// <summary>Memory does not fill up in a test; there is nothing to report.</summary>
            public string? LastError => null;

            public string? Load() => _state;

            public void Save(string state) => _state = state;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            private readonly List<int> _openedAt = [];

            public int Cycle
            {
                get; set;
            }

            public int OpenedBefore(int cycle)
            {
                var count = 0;

                foreach (var at in _openedAt)
                {
                    count += at < cycle ? 1 : 0;
                }

                return count;
            }

            public int OpenedFrom(int cycle)
            {
                var count = 0;

                foreach (var at in _openedAt)
                {
                    count += at >= cycle ? 1 : 0;
                }

                return count;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (rows[i].Kind == IncidentLogRecordKind.Incident
                        && rows[i].State == IncidentState.Opened)
                    {
                        _openedAt.Add(Cycle);
                    }
                }
            }
        }
    }
}
