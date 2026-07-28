// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// The guard on replacing <c>slopes.Sort()</c> with selection inside <see cref="TrendDetector"/>.
    ///
    /// <para><see cref="MedianSelectorTests"/> already pins the selection primitive against a sorted median.
    /// This suite asks the question one level up, which is the one that actually matters: does the shipped
    /// detector still produce the same slope and the same fitted level? The reference below re-derives
    /// Theil-Sen independently — accumulate every pairwise slope, sort the lot, take the middle — so a shared
    /// mistake between implementation and test has nowhere to hide.</para>
    ///
    /// <para>Equality is exact. Both paths reduce the same multiset of doubles to the same one or two order
    /// statistics; a tolerance here would pass a real defect.</para>
    /// </summary>
    public sealed class TrendDetectorSelectionParityTests
    {
        private const double ScrapeSeconds = 30.0;

        private static readonly TrendDetector Detector = new();

        // Both parities of the sample count, because the even case averages two order statistics and is where
        // an off-by-one hides. All are at or above TrendOptions.Balanced.MinimumSamples — below it the
        // detector returns WarmingUp with a slope of zero, and the comparison would be vacuous.
        [Theory]
        [InlineData(30)]
        [InlineData(31)]
        [InlineData(64)]
        [InlineData(121)]
        [InlineData(301)]
        public void SlopeMatchesAnIndependentSortedTheilSen(int samples)
        {
            var rng = new Random(9000 + samples);

            for (var trial = 0; trial < 12; trial++)
            {
                var (values, times) = LeakySeries(rng, samples);

                var result = Detector.Detect(values, times, TrendOptions.Balanced);
                var reference = ReferenceSlope(values, times);

                AssertDecided(result, samples);
                Assert.Equal(reference, result.SlopePerSecond);
            }
        }

        [Fact]
        public void FittedLevelMatchesToo_WhichIsWhereTheResidualMedianShows()
        {
            // The slope alone would not catch a mistake in the second selection: the intercept is the median
            // of the detrended residuals, and it reaches the caller only through FittedValueAtEnd.
            var rng = new Random(31415);

            for (var trial = 0; trial < 40; trial++)
            {
                var samples = rng.Next(TrendOptions.Balanced.MinimumSamples, 200);
                var (values, times) = LeakySeries(rng, samples);

                var result = Detector.Detect(values, times, TrendOptions.Balanced);

                var slope = ReferenceSlope(values, times);
                var intercept = ReferenceIntercept(values, times, slope);

                AssertDecided(result, samples);
                Assert.Equal(slope, result.SlopePerSecond);
                Assert.Equal(intercept + (slope * times[samples - 1]), result.FittedValueAtEnd);
            }
        }

        [Fact]
        public void AFlatSeries_IsTheDegenerateCaseAndStillMatches()
        {
            // Every pairwise slope is exactly zero: maximal ties, which is the input selection is supposed to
            // handle worst.
            var values = new double[80];
            var times = new double[80];

            for (var i = 0; i < values.Length; i++)
            {
                times[i] = i * ScrapeSeconds;
                values[i] = 512.0;
            }

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(ReferenceSlope(values, times), result.SlopePerSecond);
            Assert.Equal(0.0, result.SlopePerSecond);
            Assert.Equal(DetectionStatus.Healthy, result.Status);
        }

        [Fact]
        public void AMonotoneSeriesWithNoNoise_IsTheOtherDegenerateCase()
        {
            // Perfectly ordered input: the pivot rule's worst arrangement, and a shape that arrives in real
            // data every time a counter climbs cleanly.
            var values = new double[121];
            var times = new double[121];

            for (var i = 0; i < values.Length; i++)
            {
                times[i] = i * ScrapeSeconds;
                values[i] = 1000.0 + (i * 3.5);
            }

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(ReferenceSlope(values, times), result.SlopePerSecond);
            Assert.Equal(3.5 / ScrapeSeconds, result.SlopePerSecond, 12);
        }

        [Fact]
        public void SpikesDoNotMoveTheSlope_WhichIsTheWholePointOfTheEstimator()
        {
            // Parity is necessary but not sufficient: both paths could agree on a wrong answer. This pins the
            // property the estimator was chosen for, so a future "optimisation" cannot trade it away silently.
            var clean = new double[100];
            var times = new double[100];

            for (var i = 0; i < clean.Length; i++)
            {
                times[i] = i * ScrapeSeconds;
                clean[i] = 400.0 + (i * 2.0);
            }

            var spiked = (double[])clean.Clone();
            spiked[13] = 9e6;
            spiked[57] = -4e6;
            spiked[88] = 7e6;

            var before = Detector.Detect(clean, times, TrendOptions.Balanced).SlopePerSecond;
            var after = Detector.Detect(spiked, times, TrendOptions.Balanced).SlopePerSecond;

            Assert.Equal(before, after, 12);
        }

        /// <summary>
        /// Guards the premise. A detector that bailed out returns a slope of zero, and a reference that also
        /// happened to be near zero would let the comparison pass while measuring nothing — which is exactly
        /// how this suite first went green-adjacent on windows below the minimum.
        /// </summary>
        private static void AssertDecided(TrendResult result, int samples)
        {
            Assert.True(
                result.IsDecided,
                $"detector returned {result.Status} on {samples} samples, so the parity comparison would be "
                + $"vacuous: {result.Reason}");
        }

        private static (double[] Values, double[] Times) LeakySeries(Random rng, int samples)
        {
            var values = new double[samples];
            var times = new double[samples];

            var level = 300.0 + (rng.NextDouble() * 100.0);
            var drift = (rng.NextDouble() - 0.3) * 4.0;

            for (var i = 0; i < samples; i++)
            {
                times[i] = i * ScrapeSeconds;
                level += drift;
                values[i] = level * (1.0 + ((rng.NextDouble() - 0.5) * 0.02));

                if (rng.NextDouble() < 0.03)
                {
                    values[i] *= 5.0;
                }
            }

            return (values, times);
        }

        /// <summary>Theil-Sen the slow, obvious way: every pairwise slope, sorted, middle one.</summary>
        private static double ReferenceSlope(double[] values, double[] times)
        {
            var n = values.Length;
            var slopes = new double[n * (n - 1) / 2];
            var written = 0;

            for (var i = 0; i < n - 1; i++)
            {
                for (var j = i + 1; j < n; j++)
                {
                    slopes[written] = (values[j] - values[i]) / (times[j] - times[i]);
                    written++;
                }
            }

            Array.Sort(slopes);

            return Median(slopes);
        }

        private static double ReferenceIntercept(double[] values, double[] times, double slope)
        {
            var residuals = new double[values.Length];

            for (var i = 0; i < values.Length; i++)
            {
                residuals[i] = values[i] - (slope * times[i]);
            }

            Array.Sort(residuals);

            return Median(residuals);
        }

        private static double Median(double[] sorted)
        {
            var middle = sorted.Length / 2;

            if (sorted.Length % 2 == 1)
            {
                return sorted[middle];
            }

            return 0.5 * (sorted[middle - 1] + sorted[middle]);
        }
    }
}
