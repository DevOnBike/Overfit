// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Normalizers;

namespace DevOnBike.Overfit.Tests
{
    public class MinMaxNormalizerTests
    {
        private const int Precision = 4;

        [Fact]
        public void FitBatch_ShouldFindCorrectMinMax()
        {
            // ARRANGE
            float[] data1 = { 10f, 20f, 30f };
            float[] data2 = { 5f, 40f };
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
            // ARRANGE - Testujemy sygnał anomalii (domyślne zachowanie)
            float[] trainData = { 0f, 50f, 100f }; // Min=0, Max=100
            var normalizer = new MinMaxNormalizer { ClipToRange = false };
            normalizer.FitBatch(trainData);
            normalizer.Freeze();

            // Produkcja: wpadają wartości normalne i jedna anomalia (150f)
            float[] testData = { -10f, 50f, 150f }; 

            // ACT
            normalizer.TransformInPlace(testData);

            // ASSERT
            // -10f zostaje podbite do 0 (dolny clip zawsze działa)
            Assert.Equal(0f, testData[0], Precision);
            // 50f ląduje idealnie w połowie
            Assert.Equal(0.5f, testData[1], Precision);
            // 150f przebija sufit! Dajemy modelowi sygnał o 50% powyżej normy
            Assert.Equal(1.5f, testData[2], Precision); 
        }

        [Fact]
        public void TransformInPlace_WithClipToRangeTrue_ShouldStrictlyBoundToZeroOne()
        {
            // ARRANGE - Testujemy twarde granice
            float[] trainData = { 0f, 50f, 100f }; 
            var normalizer = new MinMaxNormalizer { ClipToRange = true };
            normalizer.FitBatch(trainData);
            normalizer.Freeze();

            // Produkcja
            float[] testData = { -50f, 50f, 200f }; 

            // ACT
            normalizer.TransformInPlace(testData);

            // ASSERT
            Assert.Equal(0f, testData[0], Precision); // Clip z dołu
            Assert.Equal(0.5f, testData[1], Precision);
            Assert.Equal(1.0f, testData[2], Precision); // Clip z góry! Anomalia została stłumiona.
        }

        [Fact]
        public void WithClipMax_ShouldInstantlyFreezeAndClip()
        {
            // ARRANGE - Tryb dla zdarzeń takich jak OomEventsRate
            var normalizer = MinMaxNormalizer.WithClipMax(5f);

            // Produkcja: 0 eventów, 3 eventy, 10 eventów (gigantyczna anomalia)
            float[] testData = { 0f, 3f, 10f };

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
            // ARRANGE - Tryb dla flag np. IsThrottled
            var normalizer = MinMaxNormalizer.Binary();
            float[] testData = { 0f, 1f, -5f, 5f }; // Wrzucamy poprawne flagi i śmieci

            // ACT
            normalizer.TransformInPlace(testData);

            // ASSERT
            Assert.Equal(0f, testData[0], Precision);
            Assert.Equal(1f, testData[1], Precision);
            Assert.Equal(0f, testData[2], Precision); // Ujemne śmieci stają się zerem
            Assert.Equal(1f, testData[3], Precision); // Dodatnie śmieci stają się jedynką
        }

        [Fact]
        public void Freeze_WithoutData_ShouldThrowException()
        {
            // ARRANGE
            var normalizer = new MinMaxNormalizer();

            // ACT & ASSERT - Zamrażanie bez danych jest nielegalne
            Assert.Throws<InvalidOperationException>(normalizer.Freeze);
        }
    }
}