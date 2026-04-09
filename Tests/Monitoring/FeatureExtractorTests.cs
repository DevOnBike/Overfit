// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Monitoring;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class FeatureExtractorTests
    {
        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private const float Tolerance = 1e-4f;

        /// Builds a flat row-major window from a jagged array [time][feature].
        private static float[] Window(params float[][] rows)
        {
            var featureCount = rows[0].Length;
            var result = new float[rows.Length * featureCount];

            for (var t = 0; t < rows.Length; t++)
            {
                for (var f = 0; f < featureCount; f++)
                {
                    result[t * featureCount + f] = rows[t][f];
                }
            }

            return result;
        }

        private static float[] Out(int featureCount)
            => new float[FeatureExtractor.OutputSize(featureCount)];

        private static float Mean(float[] stats, int feature = 0)
            => stats[feature * FeatureExtractor.StatsPerFeature + FeatureExtractor.MeanOffset];
        private static float Std(float[] stats, int feature = 0)
            => stats[feature * FeatureExtractor.StatsPerFeature + FeatureExtractor.StdOffset];
        private static float P95(float[] stats, int feature = 0)
            => stats[feature * FeatureExtractor.StatsPerFeature + FeatureExtractor.P95Offset];
        private static float Delta(float[] stats, int feature = 0)
            => stats[feature * FeatureExtractor.StatsPerFeature + FeatureExtractor.DeltaOffset];

        private static void AssertClose(float expected, float actual, string label)
            => Assert.True(MathF.Abs(expected - actual) <= Tolerance,
            $"{label}: expected={expected:F6}, actual={actual:F6}, diff={MathF.Abs(expected - actual):F6}");

        // -------------------------------------------------------------------------
        // OutputSize
        // -------------------------------------------------------------------------

        [Theory]
        [InlineData(1, 4)]
        [InlineData(4, 16)]
        [InlineData(8, 32)]
        public void OutputSize_WhenGivenFeatureCount_ThenReturnsFeatureCountTimesStatsPerFeature(
            int featureCount, int expected)
            => Assert.Equal(expected, FeatureExtractor.OutputSize(featureCount));

        // -------------------------------------------------------------------------
        // Extract — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Extract_WhenWindowLengthDoesNotMatchWindowSizeTimesFeatureCount_ThenThrowsArgumentException()
        {
            var window = new float[10]; // wrong size
            var output = Out(featureCount: 2);

            Assert.Throws<ArgumentException>(
            () => FeatureExtractor.Extract(window, windowSize: 3, featureCount: 2, output));
        }

        [Fact]
        public void Extract_WhenOutputBufferTooShort_ThenThrowsArgumentException()
        {
            var window = new float[6]; // 3×2
            var output = new float[3]; // needs 8

            Assert.Throws<ArgumentException>(
            () => FeatureExtractor.Extract(window, windowSize: 3, featureCount: 2, output));
        }

        // -------------------------------------------------------------------------
        // Extract — mean
        // -------------------------------------------------------------------------

        [Fact]
        public void Extract_WhenKnownValues_ThenMeanIsCorrect()
        {
            // feature 0: [1, 2, 3, 4, 5] → mean = 3.0
            var window = Window([1f], [2f], [3f], [4f], [5f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 5, featureCount: 1, output);

            AssertClose(3.0f, Mean(output), "mean");
        }

        [Fact]
        public void Extract_WhenAllValuesConstant_ThenMeanEqualsConstant()
        {
            var window = Window([7f], [7f], [7f], [7f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 4, featureCount: 1, output);

            AssertClose(7.0f, Mean(output), "mean");
        }

        [Fact]
        public void Extract_WhenWindowSize1_ThenMeanEqualsTheOnlyValue()
        {
            var window = Window([42f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 1, featureCount: 1, output);

            AssertClose(42.0f, Mean(output), "mean");
        }

        // -------------------------------------------------------------------------
        // Extract — std
        // -------------------------------------------------------------------------

        [Fact]
        public void Extract_WhenKnownValues_ThenStdIsCorrect()
        {
            // [1, 3, 5, 7] → mean=4, E[X²]=21, variance=5, std=√5≈2.23607
            var window = Window([1f], [3f], [5f], [7f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 4, featureCount: 1, output);

            AssertClose(MathF.Sqrt(5f), Std(output), "std");
        }

        [Fact]
        public void Extract_WhenAllValuesConstant_ThenStdIsZero()
        {
            var window = Window([5f], [5f], [5f], [5f], [5f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 5, featureCount: 1, output);

            AssertClose(0f, Std(output), "std");
        }

        [Fact]
        public void Extract_WhenWindowSize1_ThenStdIsZero()
        {
            var window = Window([99f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 1, featureCount: 1, output);

            AssertClose(0f, Std(output), "std");
        }

        [Fact]
        public void Extract_WhenTwoValues_ThenStdIsCorrect()
        {
            // [2, 8] → mean=5, E[X²]=34, variance=34/2-25=17-25... 
            // mean=5, E[X²]=(4+64)/2=34, variance=34-25=9 → std=3
            var window = Window([2f], [8f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 2, featureCount: 1, output);

            AssertClose(3.0f, Std(output), "std");
        }

        // -------------------------------------------------------------------------
        // Extract — p95
        // -------------------------------------------------------------------------

        [Fact]
        public void Extract_WhenKnownValues_ThenP95IsCorrect()
        {
            // [1,2,3,4,5] → ceil(0.95×5)-1 = ceil(4.75)-1 = 5-1 = 4 → sorted[4] = 5
            var window = Window([3f], [1f], [4f], [2f], [5f]); // unsorted input
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 5, featureCount: 1, output);

            AssertClose(5.0f, P95(output), "p95");
        }

        [Fact]
        public void Extract_WhenWindowSize6_ThenP95IsLastElement()
        {
            // ceil(0.95×6)-1 = ceil(5.7)-1 = 6-1 = 5 → sorted[5] = max
            var window = Window([10f], [20f], [30f], [40f], [50f], [60f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 6, featureCount: 1, output);

            AssertClose(60.0f, P95(output), "p95");
        }

        [Fact]
        public void Extract_WhenAllValuesConstant_ThenP95EqualsMean()
        {
            var window = Window([4f], [4f], [4f], [4f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 4, featureCount: 1, output);

            AssertClose(Mean(output), P95(output), "p95==mean");
        }

        [Fact]
        public void Extract_WhenWindowSize1_ThenP95EqualsTheOnlyValue()
        {
            var window = Window([13f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 1, featureCount: 1, output);

            AssertClose(13.0f, P95(output), "p95");
        }

        [Fact]
        public void Extract_WhenP95Computed_ThenOriginalWindowIsNotMutated()
        {
            // Wewnętrznie sortujemy scratch — oryginalne okno nie może być zmienione
            var window = new float[]
            {
                5f, 3f, 1f, 4f, 2f
            }; // windowSize=5, featureCount=1
            var copy = window.ToArray();
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 5, featureCount: 1, output);

            Assert.Equal(copy, window);
        }

        // -------------------------------------------------------------------------
        // Extract — delta
        // -------------------------------------------------------------------------

        [Fact]
        public void Extract_WhenKnownValues_ThenDeltaIsLastMinusFirst()
        {
            // first=10, last=50 → delta=40
            var window = Window([10f], [20f], [30f], [40f], [50f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 5, featureCount: 1, output);

            AssertClose(40.0f, Delta(output), "delta");
        }

        [Fact]
        public void Extract_WhenIncreasingTrend_ThenDeltaIsPositive()
        {
            var window = Window([1f], [2f], [3f], [4f], [5f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 5, featureCount: 1, output);

            Assert.True(Delta(output) > 0, $"delta should be positive, got {Delta(output)}");
        }

        [Fact]
        public void Extract_WhenDecreasingTrend_ThenDeltaIsNegative()
        {
            var window = Window([50f], [40f], [30f], [20f], [10f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 5, featureCount: 1, output);

            Assert.True(Delta(output) < 0, $"delta should be negative, got {Delta(output)}");
        }

        [Fact]
        public void Extract_WhenAllValuesConstant_ThenDeltaIsZero()
        {
            var window = Window([3f], [3f], [3f], [3f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 4, featureCount: 1, output);

            AssertClose(0f, Delta(output), "delta");
        }

        [Fact]
        public void Extract_WhenWindowSize1_ThenDeltaIsZero()
        {
            var window = Window([7f]);
            var output = Out(featureCount: 1);

            FeatureExtractor.Extract(window, windowSize: 1, featureCount: 1, output);

            AssertClose(0f, Delta(output), "delta");
        }

        // -------------------------------------------------------------------------
        // Extract — multiple features
        // -------------------------------------------------------------------------

        [Fact]
        public void Extract_WhenMultipleFeatures_ThenEachColumnComputedIndependently()
        {
            // windowSize=3, featureCount=2
            // Feature 0: [1, 2, 3] → mean=2
            // Feature 1: [10, 20, 30] → mean=20
            var window = Window([1f, 10f], [2f, 20f], [3f, 30f]);
            var output = Out(featureCount: 2);

            FeatureExtractor.Extract(window, windowSize: 3, featureCount: 2, output);

            AssertClose(2.0f, Mean(output, feature: 0), "f0 mean");
            AssertClose(20.0f, Mean(output, feature: 1), "f1 mean");
        }

        [Fact]
        public void Extract_WhenMultipleFeatures_ThenDeltaPerFeatureIsCorrect()
        {
            // Feature 0: [1→3] delta=+2   Feature 1: [30→10] delta=-20
            var window = Window([1f, 30f], [2f, 20f], [3f, 10f]);
            var output = Out(featureCount: 2);

            FeatureExtractor.Extract(window, windowSize: 3, featureCount: 2, output);

            AssertClose(2.0f, Delta(output, feature: 0), "f0 delta");
            AssertClose(-20.0f, Delta(output, feature: 1), "f1 delta");
        }

        [Fact]
        public void Extract_WhenMultipleFeatures_ThenOutputBlocksDoNotCross()
        {
            // Sprawdza że bloki stats różnych cech nie zachodzą na siebie w output
            var window = Window([1f, 100f], [2f, 200f], [3f, 300f]);
            var output = Out(featureCount: 2);

            FeatureExtractor.Extract(window, windowSize: 3, featureCount: 2, output);

            // Feature 0 max = 3, Feature 1 min = 100 — żadna stat f0 nie może być ≥ 10
            Assert.True(Mean(output, 0) < 10f, "f0 stats should not contain f1 values");
            Assert.True(Mean(output, 1) >= 10f, "f1 stats should not contain f0 values");
        }

        [Fact]
        public void Extract_WhenEightFeaturesProductionConfig_ThenOutputSizeIs32()
        {
            // Konfiguracja produkcyjna: MetricSnapshot.FeatureCount=8
            const int featureCount = 8;
            const int windowSize = 6;

            var window = new float[windowSize * featureCount];
            // Wypełnij rosnącymi wartościami
            for (var i = 0; i < window.Length; i++) { window[i] = i + 1f; }

            var output = Out(featureCount);

            // Nie rzuca — weryfikacja że cały pipeline 8-cech przechodzi
            FeatureExtractor.Extract(window, windowSize, featureCount, output);

            Assert.Equal(32, output.Length);
            // Wszystkie statystyki muszą być skończone
            foreach (var v in output)
            {
                Assert.True(float.IsFinite(v), $"non-finite value in output: {v}");
            }
        }

        // -------------------------------------------------------------------------
        // TryExtract — integracja z SlidingWindowBuffer
        // -------------------------------------------------------------------------

        [Fact]
        public void TryExtract_WhenBufferNotYetFull_ThenReturnsFalse()
        {
            using var buffer = new SlidingWindowBuffer(windowSize: 3, featureCount: 1);
            var windowScratch = new float[buffer.WindowFloats];
            var output = new float[FeatureExtractor.OutputSize(buffer.FeatureCount)];

            buffer.Add(stackalloc float[]
            {
                1f
            }, DateTime.UtcNow);
            buffer.Add(stackalloc float[]
            {
                2f
            }, DateTime.UtcNow);

            var result = FeatureExtractor.TryExtract(buffer, windowScratch, output, out _);

            Assert.False(result);
        }

        [Fact]
        public void TryExtract_WhenBufferFull_ThenReturnsTrueAndFillsOutput()
        {
            using var buffer = new SlidingWindowBuffer(windowSize: 3, featureCount: 1);
            var windowScratch = new float[buffer.WindowFloats];
            var output = new float[FeatureExtractor.OutputSize(buffer.FeatureCount)];

            buffer.Add(stackalloc float[]
            {
                1f
            }, DateTime.UtcNow);
            buffer.Add(stackalloc float[]
            {
                2f
            }, DateTime.UtcNow);
            buffer.Add(stackalloc float[]
            {
                3f
            }, DateTime.UtcNow);

            var result = FeatureExtractor.TryExtract(buffer, windowScratch, output, out var windowEnd);

            Assert.True(result);
            Assert.NotEqual(default, windowEnd);
            AssertClose(2.0f, Mean(output), "mean after TryExtract");
        }

        [Fact]
        public void TryExtract_WhenCalledTwiceWithoutNewData_ThenSecondCallReturnsFalse()
        {
            using var buffer = new SlidingWindowBuffer(windowSize: 2, featureCount: 1);
            var windowScratch = new float[buffer.WindowFloats];
            var output = new float[FeatureExtractor.OutputSize(buffer.FeatureCount)];

            buffer.Add(stackalloc float[]
            {
                1f
            }, DateTime.UtcNow);
            buffer.Add(stackalloc float[]
            {
                2f
            }, DateTime.UtcNow);

            FeatureExtractor.TryExtract(buffer, windowScratch, output, out _);
            var second = FeatureExtractor.TryExtract(buffer, windowScratch, output, out _);

            Assert.False(second);
        }
    }
}