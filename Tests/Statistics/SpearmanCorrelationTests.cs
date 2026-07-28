// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    public sealed class SpearmanCorrelationTests
    {
        [Fact]
        public void PerfectlyMonotoneButNonLinear_ScoresOne()
        {
            // The reason this namespace is rank-based: y = x^3 is a perfect monotone relationship that
            // Pearson would score well below 1.0.
            var x = new double[32];
            var y = new double[32];

            for (var i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = Math.Pow(i, 3.0);
            }

            var result = SpearmanCorrelation.Correlate(x, y);

            Assert.True(result.IsUsable);
            Assert.Equal(1.0, result.Rho, 10);
            Assert.True(result.PValue < 1e-6);
        }

        [Fact]
        public void PerfectlyInverse_ScoresMinusOne()
        {
            var x = new double[24];
            var y = new double[24];

            for (var i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = -i;
            }

            var result = SpearmanCorrelation.Correlate(x, y);

            Assert.Equal(-1.0, result.Rho, 10);
            Assert.Equal(1.0, result.Strength, 10);
        }

        [Fact]
        public void OneWildOutlier_CostsARankPositionRatherThanTheCoefficient()
        {
            // The property the whole choice of a rank method rests on, checked against an analytic value
            // rather than a guessed threshold. Moving y[7] to the top of the order makes its rank 40 instead
            // of 8 and shifts the 32 values above it down one place, so
            //   sum d^2 = 32^2 + 32*1^2 = 1056  and  rho = 1 - 6*1056 / (40 * 1599) = 0.90094...
            // That is the entire damage a single arbitrarily large spike can do at n = 40.
            var x = new double[40];
            var clean = new double[40];

            for (var i = 0; i < x.Length; i++)
            {
                x[i] = i;
                clean[i] = i * 2.0;
            }

            var spiked = (double[])clean.Clone();
            spiked[7] = 1e9;

            Assert.Equal(1.0, SpearmanCorrelation.Correlate(x, clean).Rho, 10);
            Assert.Equal(1.0 - (6.0 * 1056.0 / (40.0 * 1599.0)), SpearmanCorrelation.Correlate(x, spiked).Rho, 10);

            // And the comparison that justifies the choice: the same spike takes Pearson off a cliff, because
            // its influence grows with the squared deviation instead of being capped at one rank position.
            var pearson = Pearson(x, spiked);

            Assert.True(
                Math.Abs(pearson) < 0.5,
                $"Pearson survived the spike at {pearson} — the premise of this test no longer holds");
        }

        [Fact]
        public void IndependentSeries_AreNotSignificant()
        {
            var rng = new Random(20260728);
            var x = new double[64];
            var y = new double[64];

            for (var i = 0; i < x.Length; i++)
            {
                x[i] = rng.NextDouble();
                y[i] = rng.NextDouble();
            }

            var result = SpearmanCorrelation.Correlate(x, y);

            Assert.False(result.IsSignificant(0.8, 0.01));
        }

        [Fact]
        public void FlatSeries_IsUndecidableRatherThanZero()
        {
            var x = new double[20];
            var flat = new double[20];

            for (var i = 0; i < x.Length; i++)
            {
                x[i] = i;
                flat[i] = 5.0;
            }

            var result = SpearmanCorrelation.Correlate(x, flat);

            Assert.False(result.IsUsable);
            Assert.False(result.IsSignificant(0.0, 1.0));
        }

        [Fact]
        public void NonFiniteSamples_AreDroppedPairwise()
        {
            var x = new double[30];
            var y = new double[30];

            for (var i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = i * 3.0;
            }

            y[4] = double.NaN;
            x[11] = double.PositiveInfinity;

            var result = SpearmanCorrelation.Correlate(x, y);

            Assert.Equal(28, result.SampleCount);
            Assert.Equal(1.0, result.Rho, 10);
        }

        [Fact]
        public void TooFewOverlappingSamples_IsUndecidable()
        {
            var x = new double[SpearmanCorrelation.MinimumSamples - 1];
            var y = new double[x.Length];

            for (var i = 0; i < x.Length; i++)
            {
                x[i] = i;
                y[i] = i;
            }

            Assert.False(SpearmanCorrelation.Correlate(x, y).IsUsable);
        }

        [Fact]
        public void MisalignedSeries_Throws()
        {
            Assert.Throws<ArgumentException>(
                () => SpearmanCorrelation.Correlate(new double[8], new double[9]));
        }

        [Theory]
        [InlineData(3)]
        [InlineData(6)]
        public void LagScan_RecoversTheShift(int shift)
        {
            // A leading series and its delayed echo: the scan should report the delay, and the sign should say
            // which one led. Noise is added so the answer cannot come from an exact match.
            var rng = new Random(4711);
            var leader = new double[80];
            var follower = new double[80];

            for (var i = 0; i < leader.Length; i++)
            {
                leader[i] = Math.Sin(i * 0.18) + (rng.NextDouble() * 0.02);
            }

            for (var i = 0; i < follower.Length; i++)
            {
                var source = i - shift;
                follower[i] = source >= 0 ? leader[source] : leader[0];
            }

            var result = SpearmanCorrelation.CorrelateWithLag(leader, follower, 10);

            Assert.Equal(shift, result.LagSamples);
            Assert.True(result.Strength > 0.95, $"strength {result.Strength}");
        }

        [Fact]
        public void LagScan_PenalisesTheMultipleComparisons()
        {
            // Same data, same best lag; the only difference is how many offsets were allowed to compete. The
            // scanned p-value must be the larger one, or the scan is quietly buying significance.
            //
            // The noise is heavy on purpose. A near-perfect relationship drives 1 - CDF(|z|) below the
            // smallest positive double, and 0 x 17 is still 0 — the correction would be applied and invisible,
            // and the test would pass or fail for reasons having nothing to do with it.
            var rng = new Random(9090);
            var a = new double[40];
            var b = new double[40];

            for (var i = 0; i < a.Length; i++)
            {
                a[i] = i;
                b[i] = i + (rng.NextDouble() * 25.0);
            }

            var unscanned = SpearmanCorrelation.CorrelateWithLag(a, b, 0);
            var scanned = SpearmanCorrelation.CorrelateWithLag(a, b, 8);

            Assert.Equal(0, unscanned.LagSamples);
            Assert.True(
                unscanned.PValue > 0.0 && scanned.PValue < 1.0,
                $"premise gone: p must be strictly inside (0, 1) for the correction to be observable, "
                + $"got unscanned={unscanned.PValue}, scanned={scanned.PValue}");
            Assert.True(
                scanned.PValue > unscanned.PValue,
                $"scanned p={scanned.PValue} was not penalised against unscanned p={unscanned.PValue}");
        }

        [Fact]
        public void LagScan_OnPureNoise_StaysInsignificantAfterCorrection()
        {
            // The failure mode the correction exists for: with 21 offsets to choose from, the best of them
            // looks impressive on noise. It must not survive the correction.
            var significant = 0;

            for (var seed = 0; seed < 40; seed++)
            {
                var rng = new Random(seed);
                var a = new double[50];
                var b = new double[50];

                for (var i = 0; i < a.Length; i++)
                {
                    a[i] = rng.NextDouble();
                    b[i] = rng.NextDouble();
                }

                if (SpearmanCorrelation.CorrelateWithLag(a, b, 10).IsSignificant(0.8, 0.01))
                {
                    significant++;
                }
            }

            Assert.True(significant <= 1, $"{significant}/40 noise pairs passed the gate");
        }

        [Fact]
        public void AtZeroLag_TheScanAgreesWithTheUnscannedCoefficient()
        {
            // The scan ranks each series once over the whole window; Correlate ranks the pairwise-complete
            // pairs. On gap-free data those are the same ranking, so the two must agree exactly — this is the
            // guard that the faster sliding form did not quietly become a different statistic.
            var rng = new Random(31337);
            var a = new double[50];
            var b = new double[50];

            for (var i = 0; i < a.Length; i++)
            {
                a[i] = i + (rng.NextDouble() * 12.0);
                b[i] = (i * 0.7) + (rng.NextDouble() * 12.0);
            }

            Assert.Equal(
                SpearmanCorrelation.Correlate(a, b).Rho,
                SpearmanCorrelation.CorrelateWithLag(a, b, 0).Rho,
                12);
        }

        [Fact]
        public void TheSlidingFormAgreesWithPerWindowRankingOnTheRecoveredLag()
        {
            // Two different statistics — per-window ranking rescales inside every overlap — so their
            // coefficients differ. What must not differ is the answer to the question actually being asked:
            // which offset is the strongest.
            var rng = new Random(2024);
            var leader = new double[90];
            var follower = new double[90];

            for (var i = 0; i < leader.Length; i++)
            {
                leader[i] = Math.Sin(i * 0.19) + (rng.NextDouble() * 0.05);
            }

            for (var i = 0; i < follower.Length; i++)
            {
                var source = Math.Max(0, i - 5);
                follower[i] = (leader[source] * 0.6) + 2.0 + (rng.NextDouble() * 0.05);
            }

            Assert.Equal(5, SpearmanCorrelation.CorrelateWithLag(leader, follower, 10).LagSamples);
            Assert.Equal(5, SpearmanCorrelation.CorrelateWithLagPerWindowRanks(leader, follower, 10).LagSamples);
        }

        [Fact]
        public void LagBeyondTheCeiling_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => SpearmanCorrelation.CorrelateWithLag(new double[64], new double[64], SpearmanCorrelation.MaxLagSamples + 1));
        }

        /// <summary>Plain Pearson, present only as the foil the rank version is measured against.</summary>
        private static double Pearson(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            var n = a.Length;
            var meanA = 0.0;
            var meanB = 0.0;

            for (var i = 0; i < n; i++)
            {
                meanA += a[i];
                meanB += b[i];
            }

            meanA /= n;
            meanB /= n;

            var covariance = 0.0;
            var varianceA = 0.0;
            var varianceB = 0.0;

            for (var i = 0; i < n; i++)
            {
                var da = a[i] - meanA;
                var db = b[i] - meanB;

                covariance += da * db;
                varianceA += da * da;
                varianceB += db * db;
            }

            return covariance / Math.Sqrt(varianceA * varianceB);
        }
    }
}
