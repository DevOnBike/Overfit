// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// How much evidence each shared cluster coordinate contributes when deciding whether two findings describe
    /// one event. Each weight is on 0…1, and zero means "this coordinate carries no information here".
    ///
    /// <para><b>These were private constants until a single-node cluster showed why they cannot be.</b> The
    /// repository's own lab runs every replica on one node, so <see cref="SameNode"/> links <i>everything to
    /// everything</i> there — the coordinate is pure noise, and there was no way to say so without a rebuild.
    /// The opposite case is just as real: on a multi-tenant cluster a shared namespace is weak evidence, while
    /// on a single-tenant one it is nearly as strong as a shared workload.</para>
    ///
    /// <para>The scores are not probabilities and do not need to sum to anything. They are multiplied by the
    /// temporal score and compared against <see cref="IncidentGroupingOptions.MinRelatedness"/>, so what matters
    /// is their ordering and their distance from that threshold.</para>
    /// </summary>
    /// <param name="SamePod">Two findings about the same process — the strongest link there is.</param>
    /// <param name="SameReplicaSet">
    /// Different pods of the same ReplicaSet — same workload <i>and</i> same version.
    ///
    /// <para>Sits above <paramref name="SameWorkload"/> because during a rollout those are different claims:
    /// same-ReplicaSet means the same software, while same-workload spans the old and the new build. Lowering
    /// <paramref name="SameWorkload"/> while leaving this one high is how an operator keeps a rollout's two
    /// halves from merging into one incident.</para>
    /// </param>
    /// <param name="SameWorkload">Replicas of one Deployment/StatefulSet, <b>possibly of different versions</b>.</param>
    /// <param name="SameNode">Unrelated workloads sharing failing hardware or a saturated kubelet. <b>Set to 0
    /// on a single-node cluster</b>, where it would otherwise relate every finding to every other.</param>
    /// <param name="SameNamespace">Weakest of the topological links; on its own it should not reach
    /// <see cref="IncidentGroupingOptions.MinRelatedness"/>.</param>
    /// <param name="Correlated">Weight a strong lagged rank correlation stands in for, when topology cannot
    /// reach the pair at all. Deliberately not as high as <paramref name="SamePod"/>: series moving together is
    /// good evidence of a shared cause, but weaker than two findings being about literally the same process.</param>
    public readonly record struct TopologyWeights(
        double SamePod,
        double SameReplicaSet,
        double SameWorkload,
        double SameNode,
        double SameNamespace,
        double Correlated)
    {
        /// <summary>
        /// The shipping defaults. Ordering is the load-bearing part: pod above workload above node above
        /// namespace, with namespace low enough that it cannot merge anything by itself under
        /// <see cref="IncidentGroupingOptions.Balanced"/>.
        /// </summary>
        public static TopologyWeights Default
        {
            get;
        } = new(
            SamePod: 1.0,
            SameReplicaSet: 0.8,
            SameWorkload: 0.7,
            SameNode: 0.6,
            SameNamespace: 0.25,
            Correlated: 0.7);

        /// <summary>
        /// For a cluster where every pod shares one node — Docker Desktop, kind, minikube, or any single-node
        /// lab. <see cref="SameNode"/> is zeroed because there it is a constant, and a constant is not evidence.
        /// </summary>
        public static TopologyWeights SingleNode { get; } = Default with { SameNode = 0.0 };

        /// <summary>
        /// Whether these weights are usable. Zero is allowed — it is how a coordinate is switched off — but a
        /// negative or above-one weight is not a weakening of the evidence, it is a mistake.
        /// </summary>
        public bool IsValid
            => InRange(SamePod)
               && InRange(SameReplicaSet)
               && InRange(SameWorkload)
               && InRange(SameNode)
               && InRange(SameNamespace)
               && InRange(Correlated)
               && SamePod > 0.0;

        private static bool InRange(double weight) => weight >= 0.0 && weight <= 1.0;
    }
}
