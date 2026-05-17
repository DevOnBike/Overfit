// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Normalizers;

namespace DevOnBike.Overfit.Tests.Preprocessing.Normalizers
{
    public class MinMaxNormalizerTests
    {
        private const int Precision = 4;

        [Fact]
        public void FitBatch_ShouldFindCorrectMinMax()
        {
            // ARRANGE
            float[] data1 = [10f, 20f, 30f];
            float[] data2 = [5f, 40f];
            var normalizer = new MinMaxNormalizer();

            // ACT
            normalizer.FitBatch(data1);
            normalizer.FitBatch(data2);
            normalizer.Freeze();

            // ASSERT
            Assert.Equal(5f, normalizer.FrozenMin);
            Assert.Equal(40f, normalizer.FrozenMax);
        }

        [Fact]
        public void TransformInPlace_WithClipToRangeFalse_ShouldAllowValuesAboveOne()
        {
            // ARRANGE - Testing anomaly signal (default behaviour)
            float[] trainData = [0f, 50f, 100f]; // Min=0, Max=100
            var normalizer = new MinMaxNormalizer { ClipToRange = false };
            normalizer.FitBatch(trainData);
            normalizer.Freeze();

            // Production: normal values and one anomaly (150f) come in
            float[] testData = [-10f, 50f, 150f];

            // ACT
            normalizer.TransformInPlace(testData);

            // ASSERT
            // -10f is raised to 0 (lower clip always applies)
            Assert.Equal(0f, testData[0], Precision);
            // 50f lands exactly at the midpoint
            Assert.Equal(0.5f, testData[1], Precision);
            // 150f breaks the ceiling! We give the model a signal 50% above the normal range
            Assert.Equal(1.5f, testData[2], Precision);
        }

        [Fact]
        public void TransformInPlace_WithClipToRangeTrue_ShouldStrictlyBoundToZeroOne()
        {
            // ARRANGE - Testujemy twarde granice
            float[] trainData = [0f, 50f, 100f];
            var normalizer = new MinMaxNormalizer { ClipToRange = true };
            normalizer.FitBatch(trainData);
            normalizer.Freeze();

            // Produkcja
            float[] testData = [-50f, 50f, 200f];

            // ACT
            normalizer.TransformInPlace(testData);

            // ASSERT
            Assert.Equal(0f, testData[0], Precision); // Clip from below
            Assert.Equal(0.5f, testData[1], Precision);
            Assert.Equal(1.0f, testData[2], Precision); // Clip from above! Anomaly was suppressed.
        }

        [Fact]
        public void WithClipMax_ShouldInstantlyFreezeAndClip()
        {
            // ARRANGE - Mode for events such as OomEventsRate
            var normalizer = MinMaxNormalizer.WithClipMax(5f);

            // Production: 0 events, 3 events, 10 events (massive anomaly)
            float[] testData = [0f, 3f, 10f];

            // ACT - Nie potrzebujemy Fit/Freeze!
            normalizer.TransformInPlace(testData);

            // ASSERT
            Assert.True(normalizer.IsFrozen);
            Assert.Equal(0f, testData[0], Precision);
            Assert.Equal(0.6f, testData[1], Precision); // 3/5 = 0.6
            Assert.Equal(1.0f, testData[2], Precision); // 10 ucina do 5, 5/5 = 1.0
        }

        [Fact]
        public void Binary_ShouldNotAlterZerosAndOnes()
        {
            // ARRANGE - Mode for flags such as IsThrottled
            var normalizer = MinMaxNormalizer.Binary();
            float[] testData = [0f, 1f, -5f, 5f]; // We insert valid flags and garbage values

            // ACT
            normalizer.TransformInPlace(testData);

            // ASSERT
            Assert.Equal(0f, testData[0], Precision);
            Assert.Equal(1f, testData[1], Precision);
            Assert.Equal(0f, testData[2], Precision); // Negative garbage becomes zero
            Assert.Equal(1f, testData[3], Precision); // Positive garbage becomes one
        }

        [Fact]
        public void Freeze_WithoutData_ShouldThrowException()
        {
            // ARRANGE
            var normalizer = new MinMaxNormalizer();

            // ACT & ASSERT - Freezing without data is illegal
            Assert.Throws<InvalidOperationException>(normalizer.Freeze);
        }
    }
}