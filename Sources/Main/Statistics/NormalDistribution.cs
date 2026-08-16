// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// The standard normal distribution, to the accuracy a decision gate can use.
    ///
    /// <para>Both rank tests in this namespace reach the normal approximation at the same point — once the
    /// sample is large enough, their discrete statistic is tested against Φ — so the function lives here rather
    /// than being duplicated per test.</para>
    /// </summary>
    public static class NormalDistribution
    {
        /// <summary>
        /// Φ(z): the probability that a standard normal variate falls at or below <paramref name="z"/>.
        /// Accurate to ~1.5e-7, which is orders of magnitude tighter than the sampling noise of any monitoring
        /// window — but it is why an exact <c>0.5</c> at <c>z = 0</c> comes back as <c>0.5000000005</c>.
        /// </summary>
        public static double Cdf(double z) => 0.5 * (1.0 + Erf(z / Math.Sqrt(2.0)));

        /// <summary>The error function, via Abramowitz &amp; Stegun 7.1.26 (|error| &lt; 1.5e-7 everywhere).</summary>
        public static double Erf(double x)
        {
            const double P = 0.3275911;
            const double A1 = 0.254829592;
            const double A2 = -0.284496736;
            const double A3 = 1.421413741;
            const double A4 = -1.453152027;
            const double A5 = 1.061405429;

            var sign = x < 0.0 ? -1.0 : 1.0;
            var absolute = Math.Abs(x);

            var t = 1.0 / (1.0 + (P * absolute));
            var poly = ((((((((A5 * t) + A4) * t) + A3) * t) + A2) * t) + A1) * t;
            var value = 1.0 - (poly * Math.Exp(-absolute * absolute));

            return sign * value;
        }
    }
}
