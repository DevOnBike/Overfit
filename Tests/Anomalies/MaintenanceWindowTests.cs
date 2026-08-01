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
    /// Declaring a period abnormal on purpose.
    ///
    /// <para>A deployment <b>is</b> a level shift, and the step detector will say so at Cliff's delta 1.00 —
    /// about something the operator did five minutes ago. A tool whose first act is to page somebody about
    /// their own change is one they mute.</para>
    /// </summary>
    public sealed class MaintenanceWindowTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        [Fact]
        public void OutsideAWindowNothingIsSuppressed()
        {
            var sink = new CapturingSink();

            Guard(sink, windows: []).RunCycle(Broken(), T0.AddMinutes(20));

            Assert.NotEmpty(sink.Rows);
            Assert.DoesNotContain(sink.Rows, r => r.IsSuppressed);
        }

        /// <summary>
        /// Marked, not dropped. An operator looking at a failed deploy wants to know what the guard saw during
        /// it; deleting the evidence to keep the log tidy removes exactly the record they came for.
        /// </summary>
        [Fact]
        public void InsideAWindowFindingsAreStillReportedAndFlagged()
        {
            var sink = new CapturingSink();
            var windows = new[]
            {
                new MaintenanceWindow(T0, T0.AddHours(1), "svc", "rolling out 4.2.0"),
            };

            Guard(sink, windows).RunCycle(Broken(), T0.AddMinutes(20));

            Assert.NotEmpty(sink.Rows);
            Assert.All(sink.Rows, r => Assert.True(r.IsSuppressed));
            Assert.All(sink.Rows, r => Assert.Equal("rolling out 4.2.0", r.SuppressedBy));
        }

        [Fact]
        public void AWindowForAnotherWorkloadDoesNotCoverThisOne()
        {
            var sink = new CapturingSink();
            var windows = new[] { new MaintenanceWindow(T0, T0.AddHours(1), "other-svc", "their deploy") };

            Guard(sink, windows).RunCycle(Broken(), T0.AddMinutes(20));

            Assert.DoesNotContain(sink.Rows, r => r.IsSuppressed);
        }

        [Fact]
        public void AnUnnamedWindowStillSaysOneExisted()
        {
            var sink = new CapturingSink();

            Guard(sink, [new MaintenanceWindow(T0, T0.AddHours(1))]).RunCycle(Broken(), T0.AddMinutes(20));

            // "Suppressed" with no explanation is indistinguishable from a bug six weeks later.
            Assert.All(sink.Rows, r => Assert.NotEqual(string.Empty, r.SuppressedBy));
        }

        /// <summary>
        /// <b>The half that is easy to forget.</b> The calibrator's one real hazard is that a fault inside the
        /// observed period raises the floor above that fault and blinds the guard to it permanently. A
        /// declared window is a period already known to be abnormal, so learning from it takes the one input
        /// that is certainly wrong and treats it as ground truth.
        /// </summary>
        [Fact]
        public void NothingObservedInsideAWindowReachesTheFloors()
        {
            var learned = new CapturingSink();
            var suppressed = new CapturingSink();

            var learning = Guard(learned, windows: []);
            var notLearning = Guard(suppressed, [new MaintenanceWindow(T0, T0.AddDays(1), "svc", "load test")]);

            for (var cycle = 0; cycle < 40; cycle++)
            {
                learning.RunCycle(Broken(cycle), T0.AddMinutes(5 * cycle));
                notLearning.RunCycle(Broken(cycle), T0.AddMinutes(5 * cycle));
            }

            Assert.True(learning.FloorProposals[(int)MetricIndex.GcGen2HeapBytes].IsUsable);
            Assert.False(notLearning.FloorProposals[(int)MetricIndex.GcGen2HeapBytes].IsUsable,
                "the declared-abnormal period was folded into the baseline anyway");
        }

        private static AnomalyGuard Guard(CapturingSink sink, IReadOnlyList<MaintenanceWindow> windows)
        {
            return new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = "svc",
                    MaintenanceWindows = windows,
                    MinimumHistoryDays = 0,
                    ApplyCalibratedFloors = false,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);
        }

        /// <summary>Four replicas, one of which is plainly different — something is always found here.</summary>
        private static MetricWindow Broken(int cycle = 0)
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0.AddMinutes(5 * cycle), TimeSpan.FromSeconds(15));
            var rng = new Random(20260801 + cycle);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var heap = window.Series(pod, MetricIndex.GcGen2HeapBytes);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);
                var level = pod == 3 ? 3.0e8 : 4.0e7;

                for (var i = 0; i < window.Length; i++)
                {
                    heap[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));
                    rps[i] = 6.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));
                }
            }

            return window;
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<IncidentLogRecord> Rows { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Rows.Add(rows[i]);
                }
            }
        }
    }
}
