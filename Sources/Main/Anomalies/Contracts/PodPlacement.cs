// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>Where a pod sits in the cluster: what owns it, and what it runs on.</summary>
    /// <param name="Workload">Deployment or StatefulSet a human would name.</param>
    /// <param name="ReplicaSet">
    /// The version, on a Deployment. Distinct from the workload because during a rollout two ReplicaSets are
    /// serving at once, and a peer comparison across them compares <b>two different builds</b> — which is why
    /// the grouper scores <c>SameReplicaSet</c> above <c>SameWorkload</c>.
    /// </param>
    /// <param name="Node">Node the pod is scheduled on; empty when unknown.</param>
    /// <param name="PeerGroup">
    /// Which replicas this one may be compared against, taken from a pod label the operator names.
    ///
    /// <para><b>Declared, because it cannot be inferred.</b> Three situations produce the same shape — a
    /// minority of pods behaving unlike the majority — and call for opposite answers. A <b>rollout</b> should
    /// not be compared across, because the new cohort is starting cold. A <b>canary</b> should be compared
    /// across, because that is the entire purpose of running one. A <b>leader</b> should not be, because it
    /// legitimately does different work. Code cannot tell them apart; an attempt to key this on the
    /// ReplicaSet was written and reverted the same hour, when a test showed it made canaries invisible by
    /// leaving them alone in a cohort below the minimum group size.</para>
    ///
    /// <para><b>Read from the cluster rather than written by hand, which is the part that makes it work.</b>
    /// A hand-maintained list is wrong the moment leadership moves — and leadership moving is itself an event
    /// worth noticing. Operators already publish the fact as a label (Patroni sets <c>role</c>, and most
    /// database and queue operators do the same), kube-state-metrics exposes it as <c>kube_pod_labels</c>,
    /// and it changes on failover on its own. The client configures one string: which label to read.</para>
    ///
    /// <para>Empty means "no grouping declared", and every pod then compares against every other — the
    /// behaviour before this existed.</para>
    /// </param>
    /// <param name="CreatedAt">
    /// When the cluster created this pod, or <c>default</c> when the topology cannot say.
    ///
    /// <para><b>Taken from the cluster rather than counted in the guard, and the difference is a blind
    /// spot.</b> Counting cycles-since-first-seen would be simpler and would reset on every guard restart,
    /// so a rolling update of the monitoring tool would silence the trend family on every pod at once for
    /// the length of the grace — silence that looks exactly like health, which is the failure this whole
    /// subsystem exists to remove. kube-state-metrics already knows the answer and it survives anything the
    /// guard does. See <c>AnomalyGuardOptions.WarmUpGrace</c>.</para>
    /// </param>
    public readonly record struct PodPlacement(
        string Workload,
        string ReplicaSet,
        string Node,
        string PeerGroup = "",
        DateTimeOffset CreatedAt = default)
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
