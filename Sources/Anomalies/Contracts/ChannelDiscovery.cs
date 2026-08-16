// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>What discovery concluded about one channel.</summary>
    /// <param name="Metric">The channel.</param>
    /// <param name="Outcome">Whether it can be bound, and how confidently.</param>
    /// <param name="Chosen">
    /// The proposed binding, when there is exactly one evidenced candidate. Its <c>Source</c> is empty
    /// otherwise — including in the ambiguous case, because picking one of several silently is the failure
    /// this whole step exists to prevent.
    /// </param>
    /// <param name="Candidates">Everything that matched, evidenced or not, so a human can see what was rejected.</param>
    public readonly record struct ChannelDiscovery(
        MetricIndex Metric,
        DiscoveryOutcome Outcome,
        MetricCandidate Chosen,
        IReadOnlyList<MetricCandidate> Candidates);
}
