// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// Reading ownership out of kube-state-metrics.
    ///
    /// <para><b>Everything here turns on the answer being in the labels.</b> kube-state-metrics emits a
    /// constant <c>1</c> and carries the relationship in <c>owner_kind</c>, <c>owner_name</c>, <c>node</c>.
    /// Every other reader in this project parses for a value, so a reader written the usual way comes back
    /// with a column of ones and no topology at all.</para>
    /// </summary>
    public sealed class PrometheusTopologySourceTests
    {
        [Fact]
        public async Task ResolvesAPodThroughItsReplicaSetToTheDeployment()
        {
            using var source = Source(new Stub
            {
                PodOwners =
                [
                    ("api-7d9f8-abcde", "ReplicaSet", "api-7d9f8"),
                    ("api-7d9f8-fghij", "ReplicaSet", "api-7d9f8"),
                ],
                ReplicaSetOwners = [("api-7d9f8", "api")],
                PodNodes = [("api-7d9f8-abcde", "node-1"), ("api-7d9f8-fghij", "node-2")],
            });

            Assert.Equal(2, await source.RefreshAsync());
            Assert.True(source.TryResolve("api-7d9f8-abcde", out var placement));

            Assert.Equal("api", placement.Workload);
            Assert.Equal("api-7d9f8", placement.ReplicaSet);
            Assert.Equal("node-1", placement.Node);
        }

        /// <summary>
        /// The case the whole fix was about: two Deployments whose pods would otherwise be assumed to share a
        /// workload must come back with different ones.
        /// </summary>
        [Fact]
        public async Task TwoDeploymentsResolveToDifferentWorkloads()
        {
            using var source = Source(new Stub
            {
                PodOwners =
                [
                    ("srv-111-aaaaa", "ReplicaSet", "srv-111"),
                    ("srv-degraded-222-bbbbb", "ReplicaSet", "srv-degraded-222"),
                ],
                ReplicaSetOwners = [("srv-111", "srv"), ("srv-degraded-222", "srv-degraded")],
                PodNodes = [],
            });

            await source.RefreshAsync();

            Assert.True(source.TryResolve("srv-111-aaaaa", out var healthy));
            Assert.True(source.TryResolve("srv-degraded-222-bbbbb", out var degraded));

            Assert.NotEqual(healthy.Workload, degraded.Workload);
            Assert.Equal("srv", healthy.Workload);
            Assert.Equal("srv-degraded", degraded.Workload);
        }

        /// <summary>
        /// A StatefulSet pod has no ReplicaSet. Its owner already names the workload, and inventing a
        /// ReplicaSet for it would be worse than reporting what is there.
        /// </summary>
        [Fact]
        public async Task AnOwnerThatIsNotAReplicaSetIsTheWorkloadItself()
        {
            using var source = Source(new Stub
            {
                PodOwners = [("kafka-0", "StatefulSet", "kafka")],
                ReplicaSetOwners = [],
                PodNodes = [("kafka-0", "node-3")],
            });

            await source.RefreshAsync();

            Assert.True(source.TryResolve("kafka-0", out var placement));
            Assert.Equal("kafka", placement.Workload);
            Assert.Equal(string.Empty, placement.ReplicaSet);
            Assert.Equal("node-3", placement.Node);
        }

        /// <summary>
        /// An empty answer must not be adopted. Every pod would share an empty workload and merge into one
        /// incident — the failure this reader exists to prevent, arriving through the reader itself.
        /// </summary>
        [Fact]
        public async Task AnEmptyAnswerKeepsThePreviousSnapshot()
        {
            var stub = new Stub
            {
                PodOwners = [("api-1-a", "ReplicaSet", "api-1")],
                ReplicaSetOwners = [("api-1", "api")],
                PodNodes = [],
            };

            using var source = Source(stub);

            Assert.Equal(1, await source.RefreshAsync());

            stub.PodOwners = [];
            stub.ReplicaSetOwners = [];

            Assert.Equal(-1, await source.RefreshAsync());
            Assert.True(source.TryResolve("api-1-a", out var placement));
            Assert.Equal("api", placement.Workload);
        }

        /// <summary>Same reasoning for a failing Prometheus: stale beats blank.</summary>
        [Fact]
        public async Task AFailedRefreshKeepsThePreviousSnapshot()
        {
            var stub = new Stub
            {
                PodOwners = [("api-1-a", "ReplicaSet", "api-1")],
                ReplicaSetOwners = [("api-1", "api")],
                PodNodes = [],
            };

            using var source = Source(stub);
            await source.RefreshAsync();

            stub.Fail = true;

            Assert.Equal(-1, await source.RefreshAsync());
            Assert.True(source.TryResolve("api-1-a", out _));
            Assert.Equal(1, source.Count);
        }

        [Fact]
        public async Task AnUnknownPodIsNotResolved()
        {
            using var source = Source(new Stub
            {
                PodOwners = [("api-1-a", "ReplicaSet", "api-1")],
                ReplicaSetOwners = [("api-1", "api")],
                PodNodes = [],
            });

            await source.RefreshAsync();

            Assert.False(source.TryResolve("something-else", out var placement));
            Assert.False(placement.IsKnown);
        }

        private static PrometheusTopologySource Source(Stub stub)
        {
            return new PrometheusTopologySource(
                "http://127.0.0.1:9090",
                new PrometheusMetricSourceConfig
                {
                    PrometheusBaseUrl = "http://127.0.0.1:9090",
                    PodRegex = ".*",
                    Namespace = "overfit",
                    DataCenterLabel = string.Empty,
                },
                new HttpClient(stub));
        }

        /// <summary>Answers each query with the label shape kube-state-metrics actually emits.</summary>
        private sealed class Stub : HttpMessageHandler
        {
            public (string Pod, string Kind, string Owner)[] PodOwners { get; set; } = [];

            public (string ReplicaSet, string Owner)[] ReplicaSetOwners { get; set; } = [];

            public (string Pod, string Node)[] PodNodes { get; set; } = [];

            public bool Fail
            {
                get; set;
            }

            protected override Task<HttpResponseMessage> SendAsync(
                HttpRequestMessage request,
                CancellationToken cancellationToken)
            {
                if (Fail)
                {
                    return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
                }

                var query = Uri.UnescapeDataString(request.RequestUri?.Query ?? string.Empty);
                var series = new List<string>();

                if (query.Contains("kube_pod_owner", StringComparison.Ordinal))
                {
                    foreach (var (pod, kind, owner) in PodOwners)
                    {
                        series.Add($"{{\"metric\":{{\"pod\":\"{pod}\",\"owner_kind\":\"{kind}\","
                                   + $"\"owner_name\":\"{owner}\"}},\"value\":[0,\"1\"]}}");
                    }
                }
                else if (query.Contains("kube_replicaset_owner", StringComparison.Ordinal))
                {
                    foreach (var (replicaSet, owner) in ReplicaSetOwners)
                    {
                        series.Add($"{{\"metric\":{{\"replicaset\":\"{replicaSet}\","
                                   + $"\"owner_name\":\"{owner}\"}},\"value\":[0,\"1\"]}}");
                    }
                }
                else if (query.Contains("kube_pod_info", StringComparison.Ordinal))
                {
                    foreach (var (pod, node) in PodNodes)
                    {
                        series.Add($"{{\"metric\":{{\"pod\":\"{pod}\",\"node\":\"{node}\"}},"
                                   + "\"value\":[0,\"1\"]}");
                    }
                }

                var body = "{\"status\":\"success\",\"data\":{\"resultType\":\"vector\",\"result\":["
                           + string.Join(",", series) + "]}}";

                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(body),
                });
            }
        }
    }
}
