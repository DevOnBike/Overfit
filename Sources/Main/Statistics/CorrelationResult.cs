// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// How two series move together, and — when the lag scan was used — how far apart in time.
    /// </summary>
    /// <param name="Rho">Spearman's rank correlation on −1…+1, or <see cref="double.NaN"/> when the pair could
    /// not be evaluated. Rank-based for the same reason the rest of this namespace is: a scrape spike would
    /// dominate a Pearson coefficient and barely move a rank one.</param>
    /// <param name="PValue">Two-sided p-value from the Fisher z-transform. When a lag scan was run this is
    /// <b>already Bonferroni-corrected</b> for the number of lags tried — taking the best of 21 lags and
    /// reporting its raw p-value would be picking a winner and then pretending there was only ever one
    /// candidate.</param>
    /// <param name="SampleCount">Overlapping observations the coefficient rests on, after non-finite samples
    /// were dropped and the lag shift was applied.</param>
    /// <param name="LagSamples">Offset at which <paramref name="Rho"/> was strongest, in samples. Positive
    /// means the <i>first</i> series leads — its movement shows up in the second one this many samples later.
    /// Zero for <see cref="SpearmanCorrelation.Correlate"/>, which does not scan.</param>
    public readonly record struct CorrelationResult(
        double Rho,
        double PValue,
        int SampleCount,
        int LagSamples)
    {
        /// <summary>Whether a coefficient was computed at all.</summary>
        public bool IsUsable => !double.IsNaN(Rho);

        /// <summary>Strength irrespective of direction — the number a threshold is normally expressed against.</summary>
        public double Strength => double.IsNaN(Rho) ? 0.0 : Math.Abs(Rho);

        /// <summary>A pair that is both strong enough to matter and unlikely enough to be chance.</summary>
        public bool IsSignificant(double minStrength, double maxPValue)
            => IsUsable && Strength >= minStrength && PValue <= maxPValue;

        /// <summary>The empty verdict: not enough overlapping data to say anything.</summary>
        public static CorrelationResult Undecidable(int sampleCount)
            => new(double.NaN, 1.0, sampleCount, 0);
    }
}
