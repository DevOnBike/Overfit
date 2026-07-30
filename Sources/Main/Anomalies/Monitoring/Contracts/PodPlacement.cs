// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>Where a pod sits in the cluster: what owns it, and what it runs on.</summary>
    /// <param name="Workload">Deployment or StatefulSet a human would name.</param>
    /// <param name="ReplicaSet">
    /// The version, on a Deployment. Distinct from the workload because during a rollout two ReplicaSets are
    /// serving at once, and a peer comparison across them compares <b>two different builds</b> — which is why
    /// the grouper scores <c>SameReplicaSet</c> above <c>SameWorkload</c>.
    /// </param>
    /// <param name="Node">Node the pod is scheduled on; empty when unknown.</param>
    public readonly record struct PodPlacement(string Workload, string ReplicaSet, string Node)
    {
        /// <summary>
        /// Whether anything was actually resolved.
        ///
        /// <para>Null-safe on purpose. <c>default(PodPlacement)</c> — which is exactly what a failed
        /// <c>TryResolve</c> hands back — leaves all three strings <c>null</c>, so the obvious
        /// <c>Workload.Length &gt; 0</c> throws on the one path every caller is told to check.</para>
        /// </summary>
        public bool IsKnown
            => !string.IsNullOrEmpty(Workload)
               || !string.IsNullOrEmpty(ReplicaSet)
               || !string.IsNullOrEmpty(Node);
    }
}
