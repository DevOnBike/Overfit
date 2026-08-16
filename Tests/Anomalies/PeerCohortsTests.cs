// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Peer comparison within a declared cohort.
    ///
    /// <para>The negative case is the important one, and it is why this is declared rather than inferred: an
    /// earlier attempt keyed the split on the ReplicaSet, which looked right for a rollout and made canaries
    /// invisible — a canary is its own ReplicaSet, so it ended up alone in a cohort below the minimum group
    /// size and stopped being compared against the baseline it exists to be compared against.</para>
    /// </summary>
    public sealed class PeerCohortsTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 1, 12, 0, 0, TimeSpan.Zero);

        /// <summary>Nothing declared is the ordinary case and must cost nothing.</summary>
        [Fact]
        public void WithNothingDeclared_EveryPodIsInOneCohort()
        {
            var cohorts = PeerCohorts.Partition(["", "", "", ""], 4);

            Assert.Single(cohorts);
            Assert.Equal([0, 1, 2, 3], cohorts[0]);
        }

        [Fact]
        public void PodsSplitByDeclaredGroup_InOrder()
        {
            var cohorts = PeerCohorts.Partition(["follower", "leader", "follower", "follower"], 4);

            Assert.Equal(2, cohorts.Count);
            Assert.Equal([0, 2, 3], cohorts[0]);
            Assert.Equal([1], cohorts[1]);
        }

        /// <summary>
        /// <b>The leader case, end to end.</b> Three replicas of one image where one holds a lease and
        /// legitimately does different work. Without a declaration it is an outlier every cycle for ever;
        /// with one it is simply not compared against pods doing a different job.
        /// </summary>
        [Fact]
        public void ADeclaredLeaderIsNotComparedAgainstItsFollowers()
        {
            var topology = new StubTopology();

            for (var p = 0; p < 8; p++)
            {
                topology[$"pod-{p}"] = new PodPlacement("db", "db-1", "node-1", "follower");
            }

            // The leader: same image, different work, and it says so.
            topology["pod-7"] = new PodPlacement("db", "db-1", "node-1", "leader");

            var window = Window(pods: 8, degraded: 7);
            var sink = new CapturingSink();

            new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "db",
                    Workload = "db",
                    PodTopology = topology,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced).RunCycle(window, T0);

            Assert.DoesNotContain(sink.Pods, pod => pod == "pod-7");
        }

        /// <summary>
        /// And the same pod, undeclared, must still be found — otherwise the test above would pass on a
        /// detector that had simply stopped working.
        /// </summary>
        [Fact]
        public void TheSamePod_Undeclared_IsStillFound()
        {
            var topology = new StubTopology();

            for (var p = 0; p < 8; p++)
            {
                topology[$"pod-{p}"] = new PodPlacement("db", "db-1", "node-1");
            }

            var window = Window(pods: 8, degraded: 7);
            var sink = new CapturingSink();

            new AnomalyGuard(
                new AnomalyGuardOptions
                {
                    Namespace = "db",
                    Workload = "db",
                    PodTopology = topology,
                    Grouping = IncidentGroupingOptions.Balanced with { Topology = TopologyWeights.SingleNode },
                },
                sink,
                IncidentTrackingOptions.Balanced).RunCycle(window, T0);

            Assert.Contains(sink.Pods, pod => pod == "pod-7");
        }

        /// <summary>
        /// kube-state-metrics flattens a Kubernetes label key into a Prometheus label name, and getting that
        /// wrong is silent: the query succeeds, matches nothing, and every pod ends up in one cohort — which
        /// looks exactly like nobody having declared anything.
        /// </summary>
        [Theory]
        [InlineData("role", "label_role")]
        [InlineData("app.kubernetes.io/name", "label_app_kubernetes_io_name")]
        [InlineData("cluster-role", "label_cluster_role")]
        [InlineData("already_flat", "label_already_flat")]
        public void AKubernetesLabelBecomesItsPrometheusSeriesName(string label, string expected)
        {
            Assert.Equal(expected, PromqlCatalog.PodLabelSeriesName(label));
        }

        private static MetricWindow Window(int pods, int degraded)
        {
            var names = new List<string>(pods);

            for (var p = 0; p < pods; p++)
            {
                names.Add($"pod-{p}");
            }

            var window = new MetricWindow(names, 80, T0, TimeSpan.FromSeconds(15));
            var rng = new Random(20260801);

            for (var p = 0; p < pods; p++)
            {
                var slow = p == degraded;
                var latency = window.Series(p, MetricIndex.LatencyP95Ms);
                var rps = window.Series(p, MetricIndex.RequestsPerSecond);

                for (var i = 0; i < window.Length; i++)
                {
                    latency[i] = (slow ? 3000.0 : 900.0) * (1.0 + ((rng.NextDouble() - 0.5) * 0.2));
                    rps[i] = 5.0 * (1.0 + ((rng.NextDouble() - 0.5) * 0.1));
                }
            }

            return window;
        }

        private sealed class StubTopology : IPodTopology
        {
            private readonly Dictionary<string, PodPlacement> _pods = new(StringComparer.Ordinal);

            public PodPlacement this[string pod]
            {
                set => _pods[pod] = value;
            }

            public bool TryResolve(string pod, out PodPlacement placement)
                => _pods.TryGetValue(pod, out placement);
        }

        private sealed class CapturingSink : IIncidentSink
        {
            public List<string> Pods { get; } = [];

            public void Report(ReadOnlySpan<IncidentLogRecord> rows)
            {
                for (var i = 0; i < rows.Length; i++)
                {
                    if (rows[i].Pod.Length > 0)
                    {
                        Pods.Add(rows[i].Pod);
                    }
                }
            }
        }
    }
}
