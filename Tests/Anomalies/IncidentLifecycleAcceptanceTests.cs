// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The property that decides whether this is deployable: <b>one problem produces one notification and one
    /// closure, however many cycles it spans.</b>
    ///
    /// <para><b>This exists because it was violated on real data.</b> A twenty-two-cycle shadow run against
    /// the live lab — one fault introduced, then removed — opened <b>eight</b> incidents instead of two,
    /// alternating opened/ongoing every other cycle. The cause was the matching key: identity was keyed on
    /// (subject, signal) pairs, and once the big fault cleared the surviving incidents held one to three
    /// findings, at which size a single signal rotating out drops the overlap below any threshold. Every one
    /// of those eight was the same pod.</para>
    ///
    /// <para><b>Deterministic, and grounded in the recorded cluster rather than a simulator.</b> The "problem
    /// present" cycles feed the guard the real lab window, degraded replica included; the "problem gone"
    /// cycles feed the same window with that replica removed. That is the shape of the experiment that
    /// falsified the design, reduced to something that runs in milliseconds and fails a build.</para>
    /// </summary>
    public sealed class IncidentLifecycleAcceptanceTests
    {
        private const int PresentCycles = 6;
        private const int AbsentCycles = 6;

        private static readonly DateTimeOffset T0 = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);
        private static readonly TimeSpan Cadence = TimeSpan.FromMinutes(1);

        /// <summary>
        /// The acceptance criterion, and it is about <b>the fault</b>, not about everything the guard says.
        ///
        /// <para>Exactly one incident opens while the fault is present, and exactly one resolves once it is
        /// gone. A second incident does open when the fault disappears — the surviving healthy replicas still
        /// produce memory findings — and that is the known false-positive budget, measured elsewhere. Folding
        /// it into this assertion would make a lifecycle test fail for a detection reason, which is how a
        /// suite stops telling you which thing broke.</para>
        /// </summary>
        [Fact]
        public void TheFault_OpensOnce_AndResolvesOnce()
        {
            if (!LabWindowFixture.Exists)
            {
                return;
            }

            var (withFault, faulted) = LabWindowFixture.Load();
            var withoutFault = Without(withFault, faulted[0]);

            var sink = new CapturingSink();
            var guard = new AnomalyGuard(Options(), sink, IncidentTrackingOptions.Balanced);

            for (var cycle = 0; cycle < PresentCycles; cycle++)
            {
                guard.RunCycle(withFault, T0 + (Cadence * cycle));
            }

            for (var cycle = 0; cycle < AbsentCycles; cycle++)
            {
                guard.RunCycle(withoutFault, T0 + (Cadence * (PresentCycles + cycle)));
            }

            // Identify the fault's incident by its subject, then count only ITS state changes. Counting every
            // incident would fold in the healthy-replica memory noise, which legitimately opens one of its
            // own — a detection question, and folding it in here makes a lifecycle test fail for it.
            var opened = new SortedSet<long>();
            var resolved = new SortedSet<long>();

            foreach (var row in sink.IncidentRows)
            {
                if (!string.Equals(row.Pod, faulted[0], StringComparison.Ordinal))
                {
                    continue;
                }

                if (row.State == IncidentState.Opened)
                {
                    opened.Add(row.IncidentId);
                }

                if (row.State == IncidentState.Resolved)
                {
                    resolved.Add(row.IncidentId);
                }
            }

            Assert.Single(opened);
            Assert.Single(resolved);
            Assert.Equal(opened, resolved);
        }

        /// <summary>
        /// The immortal-incident shape, which is the one that nearly shipped. Keyed only on how much the
        /// subject sets overlapped, the incident about the degraded replica matched the incident about the
        /// three surviving ones — the grouper merges every pod's findings into one group, so they shared
        /// three subjects out of four and scored 0.75. It never resolved; it silently changed what it was
        /// about while keeping its identity, which is worse than opening a new one.
        /// </summary>
        [Fact]
        public void AnIncidentDoesNotSurviveTheSubjectItWasAbout()
        {
            if (!LabWindowFixture.Exists)
            {
                return;
            }

            var (withFault, faulted) = LabWindowFixture.Load();
            var withoutFault = Without(withFault, faulted[0]);

            var guard = new AnomalyGuard(Options(), new NullSink(), IncidentTrackingOptions.Balanced);

            for (var cycle = 0; cycle < PresentCycles; cycle++)
            {
                guard.RunCycle(withFault, T0 + (Cadence * cycle));
            }

            var resolved = 0;

            for (var cycle = 0; cycle < AbsentCycles; cycle++)
            {
                resolved += guard.RunCycle(withoutFault, T0 + (Cadence * (PresentCycles + cycle))).Resolved;
            }

            Assert.True(resolved > 0,
                "the incident about the degraded replica never closed after that replica was removed — it "
                + "has been matched onto findings about the pods that remain");
        }

        /// <summary>
        /// The flapping shape, pinned directly: a cycle must never both open and resolve the same problem.
        /// That combination is what the failing run showed, and it is invisible in a total.
        /// </summary>
        [Fact]
        public void NoCycleBothOpensAndResolves()
        {
            if (!LabWindowFixture.Exists)
            {
                return;
            }

            var (withFault, faulted) = LabWindowFixture.Load();
            var withoutFault = Without(withFault, faulted[0]);

            var guard = new AnomalyGuard(Options(), new NullSink(), IncidentTrackingOptions.Balanced);
            var churned = new List<int>();

            for (var cycle = 0; cycle < PresentCycles + AbsentCycles; cycle++)
            {
                var window = cycle < PresentCycles ? withFault : withoutFault;
                var result = guard.RunCycle(window, T0 + (Cadence * cycle));

                if (result.Opened > 0 && result.Resolved > 0)
                {
                    churned.Add(cycle);
                }
            }

            Assert.True(churned.Count == 0,
                $"cycles {string.Join(", ", churned)} both opened and resolved an incident — that is the "
                + "flapping the subject-keyed matching was introduced to remove");
        }

        /// <summary>
        /// The guard must not go quiet because the pod vanished: the coverage channel has to notice that a
        /// metric only that pod exported is now reported by nobody.
        /// </summary>
        [Fact]
        public void RemovingThePodMakesItsExclusiveMetricVisiblyBlind()
        {
            if (!LabWindowFixture.Exists)
            {
                return;
            }

            var (withFault, faulted) = LabWindowFixture.Load();
            var withoutFault = Without(withFault, faulted[0]);

            var guard = new AnomalyGuard(Options(), new NullSink(), IncidentTrackingOptions.Balanced);

            var before = guard.RunCycle(withFault, T0);
            var after = guard.RunCycle(withoutFault, T0 + Cadence);

            // CpuThrottleRatio exists only on containers carrying a CPU limit — in this lab, only the
            // degraded replica. With it gone, nobody reports it, and that must read as blindness rather than
            // as quiet.
            Assert.True(after.BlindMetrics > before.BlindMetrics,
                $"blind metrics did not rise when the only reporter left: {before.BlindMetrics} -> "
                + $"{after.BlindMetrics}");
        }

        private static AnomalyGuardOptions Options()
        {
            return new AnomalyGuardOptions
            {
                Namespace = "overfit",
                Workload = "overfit-server",
                Grouping = IncidentGroupingOptions.Balanced with
                {
                    Topology = TopologyWeights.SingleNode
                },
            };
        }

        /// <summary>The same window with one pod dropped — the fault being removed from the cluster.</summary>
        private static MetricWindow Without(MetricWindow window, string pod)
        {
            var kept = new List<string>(window.Pods.Count - 1);

            foreach (var name in window.Pods)
            {
                if (!string.Equals(name, pod, StringComparison.Ordinal))
                {
                    kept.Add(name);
                }
            }

            Assert.Equal(window.Pods.Count - 1, kept.Count);

            var result = new MetricWindow(kept, window.Length, window.Start, window.Step);

            for (var p = 0; p < kept.Count; p++)
            {
                var source = -1;

                for (var i = 0; i < window.Pods.Count; i++)
                {
                    if (string.Equals(window.Pods[i], kept[p], StringComparison.Ordinal))
                    {
                        source = i;

                        break;
                    }
                }

                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    window.Series(source, (MetricIndex)m).CopyTo(result.Series(p, (MetricIndex)m));
                }
            }

            return result;
        }

        private sealed class NullSink : IIncidentSink
        {
            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
            }
        }

        /// <summary>Keeps incident rows so a run can be judged by what a consumer would have seen.</summary>
        private sealed class CapturingSink : IIncidentSink
        {
            public List<IncidentLogRecord> IncidentRows { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (rows[i].Kind == IncidentLogRecordKind.Incident)
                    {
                        IncidentRows.Add(rows[i]);
                    }
                }
            }
        }
    }
}
