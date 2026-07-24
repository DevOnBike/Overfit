// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Whether a signal's magnitude can be explained by how much work its owner happened to receive. This is a
    /// precondition on peer comparison, not a hint: "eleven replicas alike, one different" silently assumes
    /// even load balancing, and sticky sessions, uneven sharding, a hot tenant or keep-alive skew all break
    /// that assumption — at which point the detector fires on <i>correct</i> behaviour.
    /// </summary>
    public enum PeerSignalKind
    {
        /// <summary>
        /// Uneven load cannot explain a deviation: restart count, OOMKilled, readiness, container terminations,
        /// node conditions. Comparable across peers as-is, and — usefully — detectable with no application
        /// instrumentation at all, straight from cAdvisor and kube-state-metrics.
        /// </summary>
        LoadIndependent = 0,

        /// <summary>
        /// Magnitude tracks throughput: memory, CPU, network bytes, connection counts. Requires
        /// <see cref="PeerSeries.Work"/> so the comparison runs on cost <i>per unit of work</i>; without it the
        /// detector reports <see cref="DetectionStatus.InsufficientData"/> rather than guessing.
        /// </summary>
        LoadSensitive = 1,
    }
}
