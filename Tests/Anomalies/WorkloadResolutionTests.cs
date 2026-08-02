// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Where the workload name comes from when configuration does not state one, and what happens when it
    /// cannot come from anywhere.
    ///
    /// <para><b>An empty workload caused two silent failures at once.</b> A maintenance window scoped to a
    /// named workload could never match, so the operator declared a window for their rollout and got paged
    /// during it anyway; and the incident tracker's subject key collapsed to <c>"namespace/"</c>, so a memory
    /// incident that closed and a CPU incident that opened were reported as one continuing problem. The lab's
    /// own logs carried the evidence for months as <c>Anomaly incident in lab/:</c> with nothing after the
    /// slash, and nobody read it as a defect.</para>
    /// </summary>
    public sealed class WorkloadResolutionTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 2, 12, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// kube-state-metrics already knows the owner of every pod and the topology already reads it, so an
        /// unconfigured workload costs no query to resolve.
        /// </summary>
        [Fact]
        public void TheWorkloadIsDerivedFromTopologyWhenNoneIsConfigured()
        {
            var sink = new CapturingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    PodTopology = new FakeTopology("checkout"),
                    MinimumHistoryDays = 0,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            guard.RunCycle(Window(degraded: true), T0);
            guard.RunCycle(Window(degraded: true), T0.AddMinutes(5));

            Assert.NotEmpty(sink.Workloads);
            Assert.All(sink.Workloads, w => Assert.Equal("checkout", w));
        }

        /// <summary>
        /// The derived name has to reach the maintenance calendar too, because that comparison is the reason
        /// an empty workload was worth chasing at all.
        /// </summary>
        [Fact]
        public void ADerivedWorkloadMatchesAWindowScopedToIt()
        {
            var sink = new CapturingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    PodTopology = new FakeTopology("checkout"),
                    MinimumHistoryDays = 0,
                    MaintenanceWindows =
                    [
                        new MaintenanceWindow(
                            T0.AddHours(-1), T0.AddHours(1), "checkout", "rolling update")
                    ],
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            guard.RunCycle(Window(degraded: true), T0);
            guard.RunCycle(Window(degraded: true), T0.AddMinutes(5));

            Assert.NotEmpty(sink.Messages);
            Assert.All(sink.Suppressions, s => Assert.Equal("rolling update", s));
        }

        /// <summary>
        /// A window naming a workload, no workload configured and nothing to derive one from is a
        /// contradiction that can be settled before the first cycle - and if it is not settled there, the
        /// symptom is being paged during your own declared maintenance, which points at the detector rather
        /// than at the configuration that caused it.
        /// </summary>
        [Fact]
        public void TheUnresolvableCombinationIsRefusedAtStartup()
        {
            var thrown = Assert.Throws<ArgumentException>(() => new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    MaintenanceWindows =
                    [
                        new MaintenanceWindow(T0, T0.AddHours(1), "checkout", "rolling update")
                    ],
                },
                new CapturingSink(),
                IncidentTrackingOptions.Balanced));

            Assert.Contains("maintenance window", thrown.Message, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>A namespace-scoped window needs no workload and must still be accepted.</summary>
        [Fact]
        public void ANamespaceScopedWindowNeedsNoWorkload()
        {
            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    MaintenanceWindows =
                    [
                        new MaintenanceWindow(T0, T0.AddHours(1), string.Empty, "cluster upgrade")
                    ],
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                new CapturingSink(),
                IncidentTrackingOptions.Balanced);

            Assert.Equal(0, guard.OpenIncidents);
        }

        /// <summary>Four replicas, one of which is using far more CPU than the rest when degraded.</summary>
        private static MetricWindow Window(bool degraded)
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260802);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);
                var level = degraded && pod == 0 ? 0.75 : 0.20;

                for (var i = 0; i < window.Length; i++)
                {
                    cpu[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.04));
                    rps[i] = 10.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.04));
                }
            }

            return window;
        }

        private sealed class FakeTopology : IPodTopology
        {
            private readonly string _workload;

            public FakeTopology(string workload)
            {
                _workload = workload;
            }

            public bool TryResolve(string pod, out PodPlacement placement)
            {
                placement = new PodPlacement(_workload, "rs-1", "node-0");

                return true;
            }
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<string> Messages { get; } = [];

            public List<string> Workloads { get; } = [];

            public List<string> Suppressions { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    Messages.Add(rows[i].Message);
                    Workloads.Add(rows[i].Workload);
                    Suppressions.Add(rows[i].SuppressedBy);
                }
            }
        }
    }
}
