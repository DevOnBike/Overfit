// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One population a single guard process watches: a namespace, the replicas inside it, and what to call
    /// them.
    ///
    /// <para><b>Two scopes in one namespace is the common case, not an edge one.</b> <c>api-*</c> and
    /// <c>worker-*</c> are different populations, and comparing a worker against an API replica is exactly the
    /// mistake peer grouping exists to prevent — so the unit here is (namespace, pod selector), never the
    /// namespace alone.</para>
    ///
    /// <para>Bindings and thresholds are deliberately NOT on this type. They are shared at file level with
    /// per-scope overrides, because most clients run one stack and repeating thirteen metric names fifty times
    /// is how a configuration file stops being read.</para>
    /// </summary>
    public sealed class GuardScopeEntry
    {
        /// <summary>Kubernetes namespace this scope watches.</summary>
        public string Namespace { get; set; } = string.Empty;

        /// <summary>Pod-name regex selecting the group inside that namespace.</summary>
        public string PodRegex { get; set; } = string.Empty;

        /// <summary>
        /// The deployment being watched. Left blank it is derived from topology, exactly as in the
        /// single-scope path.
        /// </summary>
        public string Workload { get; set; } = string.Empty;

        /// <summary>
        /// Pod label naming which replicas may be compared against each other. Per scope rather than shared,
        /// because the label that identifies a leader in a database namespace means nothing in a stateless one.
        /// </summary>
        public string PeerGroupLabel { get; set; } = string.Empty;
    }
}
