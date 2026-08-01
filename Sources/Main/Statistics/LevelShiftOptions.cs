// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Thresholds for <see cref="LevelShiftDetector"/>.
    ///
    /// <para>Deliberately the same shape as <see cref="TrendOptions"/> and
    /// <see cref="PeerOutlierOptions"/>: a significance level, a rank effect size, and <b>two</b> size gates —
    /// one proportional and one in the signal's own units. Every detector in this codebase has converged on
    /// that trio for the same measured reason, namely that a rank statistic is scale-free and therefore cannot
    /// distinguish "consistently different" from "different enough to act on".</para>
    /// </summary>
    /// <param name="MaxPValue">Significance the two halves must differ at.</param>
    /// <param name="MinEffectSize">
    /// Smallest Cliff's delta worth reporting. Note the shape this gate has to accommodate: a step exactly in
    /// the middle of the window separates the halves completely and scores <b>1.00</b>, while the same step a
    /// quarter of the way in leaves one half mixed and scores about <b>0.40</b>. The window slides, so a real
    /// shift is seen several times as it passes through — but the bar has to be low enough to catch it on the
    /// off-centre cycles too, or a shift is only ever visible for one cycle in four.
    /// </param>
    /// <param name="MinRelativeChange">
    /// How far the second half's median must sit from the first's, as a fraction of the first. Zero disables
    /// the gate.
    /// </param>
    /// <param name="MinimumSamples">
    /// Observations required in total before any verdict is given. Both halves must also be non-empty, which
    /// is a weaker statement — this is the floor that stops a short window from producing a confident answer.
    /// </param>
    /// <param name="MinAbsoluteChange">
    /// Smallest change between the halves' medians <b>in the signal's own units</b>, worth reporting. Zero
    /// disables the gate. Supplied per metric by the caller, for the same reason as
    /// <see cref="PeerOutlierOptions.MinAbsoluteGap"/>: one number cannot serve bytes, seconds, ratios and
    /// counts.
    /// </param>
    public readonly record struct LevelShiftOptions(
        double MaxPValue,
        double MinEffectSize,
        double MinRelativeChange,
        int MinimumSamples,
        double MinAbsoluteChange = 0.0)
    {
        /// <summary>
        /// Balanced defaults: 1% significance, delta 0.35, a 25% move, 30 samples.
        ///
        /// <para><b>Tighter than the other families on purpose.</b> This detector fires on a workload-level
        /// subject, so one finding speaks about the whole deployment rather than one replica, and it is
        /// looking at a quantity — the cross-peer common component — that moves for entirely ordinary reasons
        /// every day. A daily traffic curve is a level shift if you look at a short enough window. The 25%
        /// relative gate is what keeps the ordinary shape of a day out of it, and 1% significance is
        /// affordable because a genuine step separates the halves at p around 1e-14.</para>
        ///
        /// <para><b>Use this, not <c>default</c>.</b> A <c>default(LevelShiftOptions)</c> is all zeros, which
        /// would demand p ≤ 0 and accept any magnitude; the detector rejects it.</para>
        /// </summary>
        public static LevelShiftOptions Balanced => new(0.01, 0.35, 0.25, 30);

        /// <summary>Only large, unambiguous shifts.</summary>
        public static LevelShiftOptions Strict => new(0.001, 0.474, 0.50, 60);

        /// <summary>Shorter windows and a lower bar, for a lab or a demo.</summary>
        public static LevelShiftOptions FastFeedback => new(0.05, 0.33, 0.15, 15);

        /// <summary>Whether these thresholds can produce a meaningful verdict at all.</summary>
        public bool IsValid
            => MaxPValue is > 0.0 and < 1.0
               && MinEffectSize is >= 0.0 and <= 1.0
               && MinRelativeChange >= 0.0
               && MinimumSamples >= 4
               && MinAbsoluteChange >= 0.0;
    }
}
