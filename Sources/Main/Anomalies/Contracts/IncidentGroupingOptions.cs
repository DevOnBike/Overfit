// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Thresholds for <see cref="IncidentGrouper"/>. Every one of them trades the same two failure modes
    /// against each other: too loose and the whole cluster collapses into one incident that says nothing, too
    /// tight and the operator gets back the pile of separate alerts they were trying to escape.
    /// </summary>
    /// <param name="MaxSeparation">How far apart two findings' windows may sit and still be considered the
    /// same event. Relatedness decays linearly to zero across this gap. Overlapping windows always score
    /// full marks.</param>
    /// <param name="MaxIncidentSpan">Hard ceiling on an incident's total duration. This is the brake on
    /// transitive merging: A relates to B and B to C, so without a bound a slow chain of weak links can walk
    /// an incident across an entire day and swallow everything on the way.</param>
    /// <param name="MinRelatedness">Edge threshold on 0…1. Pairs below it are never merged.</param>
    /// <param name="MinCorrelation">Rank-correlation strength at which two findings carrying series are
    /// treated as sharing a cause even when topology puts them far apart. Set to 1.0 to disable the
    /// correlation step entirely and group on time and topology alone — which is what
    /// <see cref="Strict"/> does, and which is three orders of magnitude cheaper: 204 µs against 172 ms on a
    /// 256-finding batch whose topology forces every pair through the scan. Enabling it buys the links
    /// topology cannot see; it is not free and the price scales as O(n²).</param>
    /// <param name="MaxCorrelationPValue">Significance gate for that correlation, after the lag scan's
    /// Bonferroni correction.</param>
    /// <param name="MaxLagSamples">How far the lag scan may shift one series against the other. Zero disables
    /// the scan and correlates at lag 0 only.</param>
    /// <param name="Topology">How much each shared cluster coordinate is worth as evidence. Environment-specific
    /// — a single-node cluster must zero <see cref="TopologyWeights.SameNode"/>, where it would otherwise relate
    /// every finding to every other. See <see cref="TopologyWeights"/>.</param>
    public readonly record struct IncidentGroupingOptions(
        TimeSpan MaxSeparation,
        TimeSpan MaxIncidentSpan,
        double MinRelatedness,
        double MinCorrelation,
        double MaxCorrelationPValue,
        int MaxLagSamples,
        TopologyWeights Topology)
    {
        /// <summary>
        /// The default. Five minutes of separation covers detectors that evaluate on different cadences
        /// without bridging unrelated events; two hours of span is long enough for a slow leak to stay one
        /// incident and short enough that yesterday's problem is not today's.
        /// </summary>
        public static IncidentGroupingOptions Balanced
        {
            get;
        } = new(
            MaxSeparation: TimeSpan.FromMinutes(5),
            MaxIncidentSpan: TimeSpan.FromHours(2),
            MinRelatedness: 0.35,
            MinCorrelation: 0.8,
            MaxCorrelationPValue: 0.01,
            MaxLagSamples: 10,
            Topology: TopologyWeights.Default);

        /// <summary>
        /// Groups only what is hard to argue with: same pod or same workload, tightly overlapping in time,
        /// no correlation-based linking. Fewer, smaller, more defensible incidents.
        /// </summary>
        public static IncidentGroupingOptions Strict
        {
            get;
        } = new(
            MaxSeparation: TimeSpan.FromMinutes(1),
            MaxIncidentSpan: TimeSpan.FromMinutes(30),
            MinRelatedness: 0.6,
            MinCorrelation: 1.0,
            MaxCorrelationPValue: 0.001,
            MaxLagSamples: 0,
            Topology: TopologyWeights.Default);

        /// <summary>Whether the thresholds are usable — guards against <c>default</c> being passed in.</summary>
        public bool IsValid
            => MaxSeparation > TimeSpan.Zero
               && MaxIncidentSpan > TimeSpan.Zero
               && MinRelatedness > 0.0 && MinRelatedness <= 1.0
               && MinCorrelation > 0.0 && MinCorrelation <= 1.0
               && MaxCorrelationPValue > 0.0 && MaxCorrelationPValue <= 1.0
               && MaxLagSamples >= 0
               && Topology.IsValid;
    }
}
