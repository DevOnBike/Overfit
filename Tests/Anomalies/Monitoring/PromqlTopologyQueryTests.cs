// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Tests.Anomalies.Monitoring
{
    /// <summary>
    /// The topology queries that make a rollout visible without a Kubernetes list/watch client.
    ///
    /// <para>A query builder's contract <b>is</b> its output string, so these assert the generated PromQL
    /// exactly. That matters more here than usual: the last time this catalog's queries went unchecked, three
    /// defects — a hard-coded <c>dc</c> label, names from a different exporter, and a missing <c>sum by (pod)</c>
    /// — made every one of them match nothing, and Prometheus answered <c>success</c> with no series so the
    /// failure was completely silent.</para>
    /// </summary>
    public sealed class PromqlTopologyQueryTests
    {
        private static PrometheusMetricSourceConfig Lab => new()
        {
            PrometheusBaseUrl = "http://127.0.0.1:9090",
            PodRegex = "overfit-server-.*",
            Namespace = "overfit",
            DataCenterLabel = string.Empty
        };

        [Fact]
        public void PodOwnership_FiltersByPodAndNamespace_AndPinsTheOwnerKind()
        {
            // owner_kind must be pinned: a pod is also owned by a Job or a StatefulSet, and only the ReplicaSet
            // chain carries the version.
            Assert.Equal(
                "kube_pod_owner{pod=~\"overfit-server-.*\",namespace=\"overfit\",owner_kind=\"ReplicaSet\"}",
                PromqlCatalog.PodOwnershipQuery(Lab));
        }

        [Fact]
        public void ReplicaSetOwnership_CannotFilterByPod_BecauseTheSeriesHasNoPodLabel()
        {
            // The commonest way to get this wrong is to paste the pod selector everywhere. kube_replicaset_owner
            // is keyed on `replicaset`, so a pod matcher reduces it to the empty set — silently.
            var query = PromqlCatalog.ReplicaSetOwnershipQuery(Lab);

            Assert.Equal("kube_replicaset_owner{namespace=\"overfit\"}", query);
            Assert.DoesNotContain("pod=", query, StringComparison.Ordinal);
        }

        [Fact]
        public void DeploymentCreated_IsNamespaceScopedOnly()
        {
            var query = PromqlCatalog.DeploymentCreatedQuery(Lab);

            Assert.Equal("kube_deployment_created{namespace=\"overfit\"}", query);
            Assert.DoesNotContain("pod=", query, StringComparison.Ordinal);
        }

        [Fact]
        public void PodNode_CarriesThePodSelector()
        {
            Assert.Equal(
                "kube_pod_info{pod=~\"overfit-server-.*\",namespace=\"overfit\"}",
                PromqlCatalog.PodNodeQuery(Lab));
        }

        [Fact]
        public void ReplicaSetPopulation_AggregatesTheOwnerName()
        {
            // The old-versus-new split in one query: a workload with two non-zero entries is mid-rollout.
            Assert.Equal(
                "sum by (owner_name) (kube_pod_owner{pod=~\"overfit-server-.*\","
                + "namespace=\"overfit\",owner_kind=\"ReplicaSet\"})",
                PromqlCatalog.ReplicaSetPopulationQuery(Lab));
        }

        [Fact]
        public void WithNoNamespace_TheBracesAreOmittedEntirely()
        {
            // `kube_replicaset_owner{}` is valid PromQL but an empty matcher set reads like a mistake, and a
            // stray comma would be one.
            var unscoped = Lab with { Namespace = string.Empty };

            Assert.Equal("kube_replicaset_owner", PromqlCatalog.ReplicaSetOwnershipQuery(unscoped));
            Assert.Equal("kube_deployment_created", PromqlCatalog.DeploymentCreatedQuery(unscoped));
        }

        [Fact]
        public void WithADataCentreLabel_ItIsAppliedToTheNamespaceScopedQueriesToo()
        {
            var multiDc = Lab with { DataCenterLabel = "dc", DcWestLabel = "west" };

            Assert.Equal(
                "kube_replicaset_owner{namespace=\"overfit\",dc=\"west\"}",
                PromqlCatalog.ReplicaSetOwnershipQuery(multiDc));
        }

        [Fact]
        public void WithADataCentreLabelAndNoNamespace_ThereIsNoLeadingComma()
        {
            var odd = Lab with { Namespace = string.Empty, DataCenterLabel = "dc", DcWestLabel = "west" };

            Assert.Equal("kube_deployment_created{dc=\"west\"}", PromqlCatalog.DeploymentCreatedQuery(odd));
        }
    }
}
