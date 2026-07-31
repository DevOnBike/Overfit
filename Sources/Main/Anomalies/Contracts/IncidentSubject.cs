// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What a finding is about, and where it sits in the cluster. The three coordinates are what let two
    /// findings be recognised as the same event: same pod is the strongest link, same workload the next, and
    /// same node the one that catches the fault nobody thinks to look for — a saturated or failing node
    /// degrading pods that have nothing else to do with each other.
    ///
    /// <para>Every field is available from Prometheus alone (<c>kube_pod_owner</c>, <c>kube_replicaset_owner</c>,
    /// <c>kube_pod_info</c>), which is what lets the guard run without a Kubernetes list/watch client.</para>
    /// </summary>
    /// <param name="Namespace">Kubernetes namespace.</param>
    /// <param name="Workload">Owning Deployment/StatefulSet/DaemonSet — the level a human reasons at.
    /// Empty when unknown.</param>
    /// <param name="ReplicaSet">
    /// Owning ReplicaSet, which on a Deployment is <b>the version</b>. Empty when unknown.
    ///
    /// <para>This is the coordinate a rollout turns on. During one, a workload holds two ReplicaSets running
    /// different software, and two findings from opposite sides of that split are <i>not</i> as related as two
    /// from the same side — they are about different programs. Without this field there is no way to express
    /// that, which is why the peer group goes blind through a rollout and reports "no coherent norm".</para>
    /// </param>
    /// <param name="Pod">Pod name. Empty for a finding about the workload as a whole.</param>
    /// <param name="Node">Node the pod is scheduled on. Empty when unknown.</param>
    public readonly record struct IncidentSubject(
        string Namespace,
        string Workload,
        string ReplicaSet,
        string Pod,
        string Node)
    {
        /// <summary>The most specific non-empty coordinate, for display.</summary>
        public string Label
        {
            get
            {
                if (Pod.Length > 0)
                {
                    return Pod;
                }

                if (Workload.Length > 0)
                {
                    return Workload;
                }

                return Namespace;
            }
        }
    }
}
