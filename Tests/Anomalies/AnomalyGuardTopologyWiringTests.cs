// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// That topology actually reaches the grouping, and that a wrong answer does what it is supposed to.
    ///
    /// <para><b>The failure being guarded against is "registered but never consulted".</b>
    /// <c>PromqlCatalog.PodOwnershipQuery</c> sat in this codebase for a long time with nothing executing it,
    /// and a reader that is wired but never read is indistinguishable from one that works — every pod quietly
    /// falls back to the name heuristic while nothing in the logs says so.</para>
    /// </summary>
    public sealed class AnomalyGuardTopologyWiringTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 7, 30, 12, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// Pods whose names all suggest one workload, but which topology says include a canary. The guard must
        /// follow topology — this is the difference between one incident and two.
        ///
        /// <para><b>Eight pods, and the number is not arbitrary.</b> Three is enough for the peer detector to
        /// run and not enough for it to decide: with one outlier in three, that outlier is half of each other
        /// member's leave-one-out baseline, so the others cross in the opposite direction and the group comes
        /// back <c>Inconclusive</c>. The masking bound is <c>(n-k)/(n-1)</c>; at eight members one outlier is
        /// 1/7 of each baseline, comfortably under the effect-size gate. A fixture that produces no findings
        /// would pass or fail this test for a reason that has nothing to do with topology.</para>
        /// </summary>
        [Fact]
        public void TopologyOverridesTheNameHeuristic()
        {
            var pods = Pods(8);
            var topology = new StubTopology();

            for (var i = 0; i < pods.Length; i++)
            {
                topology[pods[i]] = i == pods.Length - 1
                    ? new PodPlacement("srv-canary", "srv-999", "node-1")
                    : new PodPlacement("srv", "srv-111", "node-1");
            }

            var sink = new CapturingSink();
            var guard = new AnomalyGuard(
                Options() with { PodTopology = topology }, sink, IncidentTrackingOptions.Balanced);

            guard.RunCycle(Window(pods), T0);

            var workloads = new SortedSet<string>(StringComparer.Ordinal);

            foreach (var row in sink.Rows)
            {
                if (row.NamesAPod)
                {
                    workloads.Add(row.Workload);
                }
            }

            Assert.Contains("srv-canary", workloads);
        }

        /// <summary>
        /// An unresolved pod must not be given a blank workload. A blank one is shared by every unresolved
        /// pod, so they would all merge — the failure the topology reader exists to prevent, arriving through
        /// the fallback instead.
        /// </summary>
        [Fact]
        public void AnUnresolvedPodFallsBackToTheNameHeuristic_NotToBlank()
        {
            var sink = new CapturingSink();
            var guard = new AnomalyGuard(
                Options() with { PodTopology = new StubTopology() }, sink, IncidentTrackingOptions.Balanced);

            guard.RunCycle(Window(Pods(8)), T0);

            foreach (var row in sink.Rows)
            {
                if (row.NamesAPod)
                {
                    Assert.NotEqual(string.Empty, row.Workload);
                }
            }
        }

        private static AnomalyGuardOptions Options()
        {
            return new AnomalyGuardOptions
            {
                Namespace = "overfit",
                Workload = "fallback",
                Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
            };
        }

        private static string[] Pods(int count)
        {
            var pods = new string[count];

            for (var i = 0; i < count; i++)
            {
                pods[i] = $"srv-111-pod{i:d2}";
            }

            return pods;
        }

        /// <summary>The last pod is obviously slower, so the peer detector has something to report.</summary>
        private static MetricWindow Window(string[] pods)
        {
            var window = new MetricWindow(pods, 60, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260730);

            for (var p = 0; p < pods.Length; p++)
            {
                var latency = window.Series(p, MetricIndex.LatencyP95Ms);
                var rps = window.Series(p, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    latency[i] = (p == pods.Length - 1 ? 3200.0 : 900.0)
                                 * (1.0 + ((rng.NextDouble() - 0.5) * 0.2));
                    rps[i] = 5.0;
                }
            }

            return window;
        }

        private sealed class StubTopology : Dictionary<string, PodPlacement>, IRefreshablePodTopology
        {
            public StubTopology()
                : base(StringComparer.Ordinal)
            {
            }

            public int Refreshes
            {
                get; private set;
            }

            public bool TryResolve(string pod, out PodPlacement placement)
            {
                return TryGetValue(pod, out placement);
            }

            public Task<int> RefreshAsync(CancellationToken ct = default)
            {
                Refreshes++;

                return Task.FromResult(Count);
            }
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
