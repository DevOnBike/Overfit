// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Thresholds for trend detection — the same three-part gate the rest of the family uses: significant,
    /// materially large, and backed by enough data.
    /// </summary>
    /// <param name="MaxPValue">Significance level for the Mann-Kendall test.</param>
    /// <param name="MinTau">
    /// Smallest Kendall's tau worth reporting: how <i>monotone</i> the movement is, on −1…+1. This is the
    /// consistency of the climb, not its size — a series creeping up by 1% but never once going down scores
    /// near 1.0.
    /// </param>
    /// <param name="MinRelativeChangeOverWindow">
    /// Smallest fitted change across the window, as a fraction of the series' own median — the size of the
    /// climb, in units an operator recognises. Expressed relatively so one threshold serves bytes, seconds and
    /// request counts alike. 0.10 means "the fit must move at least 10% of typical across the window".
    /// </param>
    /// <param name="MinimumSamples">Observations required before any verdict is given.</param>
    public readonly record struct TrendOptions(
        double MaxPValue,
        double MinTau,
        double MinRelativeChangeOverWindow,
        int MinimumSamples)
    {
        /// <summary>
        /// Balanced defaults: 5% significance, tau 0.30, a 10% move across the window, 30 samples.
        ///
        /// <para><b>Use this, not <c>default</c>.</b> A <c>default(TrendOptions)</c> is all zeros, which would
        /// demand p ≤ 0 and accept any magnitude; the detector rejects it.</para>
        /// </summary>
        public static TrendOptions Balanced => new(0.05, 0.30, 0.10, 30);

        /// <summary>Production: only consistent, substantial movement, over a longer window.</summary>
        public static TrendOptions Strict => new(0.01, 0.50, 0.20, 60);

        /// <summary>Staging: reacts on less data, so it demands a larger move to compensate.</summary>
        public static TrendOptions FastFeedback => new(0.05, 0.40, 0.25, 15);

        /// <summary>Whether these thresholds are usable at all — guards the all-zero <c>default</c> trap.</summary>
        public bool IsValid
            => MaxPValue > 0.0
               && MaxPValue <= 1.0
               && MinTau > 0.0
               && MinTau <= 1.0
               && MinRelativeChangeOverWindow > 0.0
               && MinimumSamples >= 3;
    }
}
