// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Thresholds for <see cref="SustainedThresholdRule"/>: how high is too high, and for how much of the
    /// window it has to stay there.
    /// </summary>
    /// <param name="Threshold">Value at or above which a sample counts as a breach, in the signal's own units.</param>
    /// <param name="MinBreachFraction">
    /// Share of usable samples that must breach before the window is called anomalous.
    ///
    /// <para><b>Persistence is the whole reason this is a rule and not a comparison.</b> A single scrape over
    /// the line is what a burst of traffic looks like; the same line held for a quarter of the window is what a
    /// misconfigured limit looks like. Without this, an absolute threshold on a bursty signal is a noise
    /// generator.</para>
    /// </param>
    /// <param name="MinimumSamples">Usable observations required before any verdict is given.</param>
    public readonly record struct SustainedThresholdOptions(
        double Threshold,
        double MinBreachFraction,
        int MinimumSamples)
    {
        /// <summary>
        /// CPU throttling: <b>5% of CFS periods throttled, held across a quarter of the window.</b>
        ///
        /// <para><b>Both numbers come from the cluster lab, and the literature figure would have missed the
        /// fault entirely.</b> Guidance commonly names 25% throttling as the point of concern. Measured on a
        /// pod limited to one core while its siblings burst to roughly 1.8, the throttled fraction peaked at
        /// <b>19.8%</b> and never once reached 25% — median 1.4%, p90 11.8%. A 25% threshold is silent on a
        /// replica whose p95 response time was 2.75x its peers'.</para>
        ///
        /// <para>What separates loaded from idle is persistence, not height. Over the loaded window 33% of
        /// samples sat at or above 5%; over a mostly-idle half hour only 13% did. A quarter of the window sits
        /// between those with roughly comparable margin on each side.</para>
        ///
        /// <para><b>Calibrated on one fault, on one single-node lab.</b> The direction and the shape are solid —
        /// throttling is bursty, and the ceiling is far lower than the usual advice assumes. The exact numbers
        /// are not: a multi-node cluster, a different limit ratio or a CPU-bound workload should be expected to
        /// move them, and no threshold should be promoted to a product default without that repeat.</para>
        /// </summary>
        public static SustainedThresholdOptions ForCpuThrottling { get; } = new(
            Threshold: 0.05,
            MinBreachFraction: 0.25,
            MinimumSamples: 20);

        /// <summary>
        /// Any non-zero reading, sustained hardly at all: the shape for counters where a single event is the
        /// finding — OOM kills, container restarts, pool rejections. One breach in twenty samples is enough,
        /// because these do not happen by accident.
        /// </summary>
        public static SustainedThresholdOptions ForRareEvent { get; } = new(
            Threshold: double.Epsilon,
            MinBreachFraction: 0.05,
            MinimumSamples: 20);

        /// <summary>Whether these thresholds are usable — guards the all-zero <c>default</c> trap.</summary>
        public bool IsValid
            => double.IsFinite(Threshold)
               && MinBreachFraction > 0.0
               && MinBreachFraction <= 1.0
               && MinimumSamples > 0;
    }
}
