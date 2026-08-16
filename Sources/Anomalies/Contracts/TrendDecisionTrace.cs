// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One trend decision, with the gates that produced it kept apart.
    ///
    /// <para><b>Why a separate record from <see cref="PeerDecisionTrace"/>.</b> That one is shaped by the
    /// peer comparison — relative gap, absolute gap, effect size — and a trend has none of those. It has a
    /// slope, a rank correlation, an autocorrelation and a floor on the change across the window, and
    /// forcing them into peer-shaped fields would mean a reader could not tell which number meant what.</para>
    ///
    /// <para><b><see cref="WarmingUp"/> is the field this was worth building for.</b> A pod inside the
    /// warm-up grace is skipped with a bare <c>continue</c> and, before this, left no evidence at all — so
    /// "no finding" and "not even looked at" were the same silence. During a rollout that is every pod.</para>
    /// </summary>
    /// <param name="Signal">Channel name, built-in or custom.</param>
    /// <param name="Pod">The pod judged.</param>
    /// <param name="Status">The verdict, or <c>InsufficientData</c> when none was reached.</param>
    /// <param name="WarmingUp">
    /// The pod was inside the warm-up grace and no test ran.
    ///
    /// <para><b>When this is true the five measured fields below are placeholder zeros, not results</b> —
    /// <paramref name="SlopePerSecond"/>, <paramref name="KendallTau"/>, <paramref name="PValue"/>,
    /// <paramref name="Autocorrelation"/> and <paramref name="SampleCount"/> are all written as 0 because no
    /// test produced any of them. A zero <paramref name="PValue"/> is the one that misleads, since read on its
    /// own it is the most significant value the field can take. <paramref name="Status"/> is
    /// <c>InsufficientData</c> in this case and is what should be read first.</para>
    /// </param>
    /// <param name="FloorOverWindow">The change the signal had to clear, in its own unit.</param>
    /// <param name="SlopePerSecond">
    /// Theil-Sen slope in the signal's own units <b>per second</b> — the median of all pairwise slopes, so a
    /// handful of spikes cannot steer it.
    ///
    /// <para><b>Not comparable with <paramref name="FloorOverWindow"/> as it stands</b>, and that is the one
    /// thing to know before reading the two together: the floor is a change across the whole window, so the
    /// comparison the gate performs is this slope multiplied by the window length in seconds. A slope that
    /// looks tiny beside the floor may still have cleared it over an hour.</para>
    /// </param>
    /// <param name="KendallTau">
    /// Monotonicity on −1…+1 — the effect size, and the answer to "how consistently", where
    /// <paramref name="SlopePerSecond"/> answers "how fast". Sign follows the direction of travel.
    /// </param>
    /// <param name="PValue">
    /// One-sided Mann-Kendall p-value in the direction observed, already discounted for
    /// <paramref name="Autocorrelation"/>. Compared against the trend options' own bar, not read on its own.
    /// </param>
    /// <param name="Autocorrelation">
    /// Lag-1 autocorrelation of the detrended series, on 0…1 — how much the significance had to be discounted.
    /// Surfaced because a value near 1 means the window carries far less independent evidence than
    /// <paramref name="SampleCount"/> suggests, which is the case where a confident-looking p-value is not.
    /// </param>
    /// <param name="SampleCount">
    /// Observations the verdict rests on, after filtering and any thinning — not the window's length in
    /// samples, which is larger whenever a scrape returned nothing.
    /// </param>
    /// <param name="HasExpectation">A seasonal expectation was subtracted before the fit.</param>
    /// <param name="Reason">The detector's own sentence, which is what an operator reads first.</param>
    public readonly record struct TrendDecisionTrace(
        string Signal,
        string Pod,
        DetectionStatus Status,
        bool WarmingUp,
        double FloorOverWindow,
        double SlopePerSecond,
        double KendallTau,
        double PValue,
        double Autocorrelation,
        int SampleCount,
        bool HasExpectation,
        string Reason);
}
