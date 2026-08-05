// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// A scope after resolution: what the guard will actually watch, with a stable name to label its
    /// telemetry and its durable state with.
    /// </summary>
    /// <param name="Namespace">Kubernetes namespace.</param>
    /// <param name="PodRegex">Pod-name regex, or empty for every pod in the namespace.</param>
    /// <param name="Workload">Declared workload, or empty to derive it from topology.</param>
    /// <param name="PeerGroupLabel">Pod label naming comparable replicas, or empty.</param>
    public readonly record struct GuardScope(
        string Namespace,
        string PodRegex,
        string Workload,
        string PeerGroupLabel)
    {
        /// <summary>
        /// The identity this scope carries into telemetry labels and durable state.
        ///
        /// <para><b>Derived, not configured, and that is the point.</b> A name a client types is a name a
        /// client can change, and changing it would orphan the scope's saved incidents and learned floors
        /// while looking like an edit to a comment. Deriving it from the two fields that define the population
        /// means the identity moves only when the thing it identifies moves.</para>
        ///
        /// <para>The namespace alone will not do: two scopes in one namespace is the ordinary case, and giving
        /// them one name would merge their trackers — the exact cross-scope contamination this whole design is
        /// arranged to prevent.</para>
        /// </summary>
        public string Name => PodRegex.Length == 0 ? Namespace : $"{Namespace}/{PodRegex}";
    }
}
