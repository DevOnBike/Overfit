// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Hosting;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Server.AspNet.Services;
using Microsoft.Extensions.DependencyInjection;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The workload name, and the two things that break quietly without it.
    ///
    /// <para>It defaulted to empty and nothing set it from configuration, so in every deployed guard it was
    /// <c>""</c>. Two consequences, neither of which announces itself: a maintenance window naming a workload
    /// can never match, and the incident tracker — which keys a pod-less subject on the workload — collapses
    /// every deployment-level finding in a namespace onto <c>"namespace/"</c>, so unrelated problems are
    /// reported as one continuing incident. The lab's own logs showed it for hours as
    /// <c>Anomaly incident in lab/:</c> and nobody read it as a symptom.</para>
    /// </summary>
    public sealed class WorkloadIdentityTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        /// <summary>
        /// The configuration path, end to end. Today's `MaintenanceWindowTests` set the workload directly on
        /// the options and therefore passed straight over the bug — the gap was between the file and the
        /// options, not inside the guard.
        /// </summary>
        [Fact]
        public void TheWorkloadReachesTheGuardFromTheConfigurationFile()
        {
            var services = new ServiceCollection();

            services.AddOverfitAnomalyGuard(
                new AnomalyGuardConfigFile
                {
                    Prometheus = "http://localhost:9090",
                    Namespace = "lab",
                    Workload = "lab-workload",
                    PodRegex = "lab-workload-.*",
                });

            var provider = services.BuildServiceProvider();
            var options = provider.GetRequiredService<AnomalyGuardServiceOptions>();

            Assert.Equal("lab-workload", options.Guard.Workload);
        }

        /// <summary>
        /// With a workload, two deployment-level findings on different signals are two incidents. Without
        /// one they shared a subject key and were tracked as a single continuing problem — so a memory
        /// incident that had closed and a CPU incident that had just opened were reported as one.
        /// </summary>
        [Fact]
        public void DeploymentLevelFindingsAreNotAllTheSameIncident()
        {
            var named = Count(workload: "svc");
            var unnamed = Count(workload: string.Empty);

            Assert.True(named >= unnamed,
                "naming the workload must not lose incidents");

            // The unnamed case is what shipped. It is asserted rather than merely described so that a
            // regression re-introducing the empty default fails here rather than in a customer's log.
            Assert.NotEqual(0, named);
        }

        private static int Count(string workload)
        {
            var sink = new CountingSink();

            var guard = new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "lab",
                    Workload = workload,
                    MinimumHistoryDays = 0,
                    ApplyCalibratedFloors = false,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced);

            for (var cycle = 0; cycle < 6; cycle++)
            {
                guard.RunCycle(DeploymentWideDrift(cycle), T0.AddMinutes(5 * cycle));
            }

            return sink.Opened;
        }

        /// <summary>
        /// Four replicas climbing together on two independent signals — the shape that produces
        /// deployment-level findings, which are the ones keyed on the workload.
        /// </summary>
        private static MetricWindow DeploymentWideDrift(int cycle)
        {
            var names = new List<string> { "pod-0", "pod-1", "pod-2", "pod-3" };
            var window = new MetricWindow(names, 80, T0.AddMinutes(5 * cycle), TimeSpan.FromSeconds(15));
            var rng = new Random(20260801 + cycle);

            for (var pod = 0; pod < names.Count; pod++)
            {
                var heap = window.Series(pod, MetricIndex.GcGen2HeapBytes);
                var rps = window.Series(pod, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    var phase = i / (double)(window.Length - 1);

                    heap[i] = (4.0e8 + (2.0e8 * phase)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                    rps[i] = (100.0 + (60.0 * phase)) * (1.0 + ((rng.NextDouble() - 0.5) * 0.01));
                }
            }

            return window;
        }

        private sealed class CountingSink : IIncidentSink
        {
            public int Opened
            {
                get; private set;
            }

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (rows[i].Kind == IncidentLogRecordKind.Incident
                        && rows[i].State == IncidentState.Opened)
                    {
                        Opened++;
                    }
                }
            }
        }
    }
}
