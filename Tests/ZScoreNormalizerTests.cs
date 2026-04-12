using DevOnBike.Overfit.Statistical;

namespace DevOnBike.Overfit.Tests
{
    public class ZScoreNormalizerTests
    {
        [Fact]
        public void FitBatch_And_FitIncremental_ShouldYieldIdenticalResults()
        {
            // ARRANGE
            float[] data = { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 10.0f, -2.0f, 0.0f };
            var batchNormalizer = new ZScoreNormalizer();
            var incrementalNormalizer = new ZScoreNormalizer();

            // ACT
            // 1. Obliczamy parametry wsadowo (SIMD)
            batchNormalizer.FitBatch(data);

            // 2. Obliczamy parametry strumieniowo (Algorytm Welforda)
            foreach (var val in data)
            {
                incrementalNormalizer.FitIncremental(val);
            }

            // ASSERT
            Assert.Equal(data.Length, batchNormalizer.Count);
            Assert.Equal(data.Length, incrementalNormalizer.Count);

            // Tolerancja 4 miejsc po przecinku (1e-4) dla różnic w operacjach zmiennoprzecinkowych
            Assert.Equal(batchNormalizer.Mean, incrementalNormalizer.Mean, precision: 4);
            Assert.Equal(batchNormalizer.StandardDeviation, incrementalNormalizer.StandardDeviation, precision: 4);
        }

        [Fact]
        public void TransformInPlace_ShouldCorrectlyStandardizeData()
        {
            // ARRANGE
            // Znany zbiór danych, dla którego łatwo policzyć parametry w pamięci:
            // Średnia = 5.0
            // Wariancja = 4.0 -> Odchylenie Standardowe (StdDev) = 2.0
            float[] data = { 2f, 4f, 4f, 4f, 5f, 5f, 7f, 9f };

            var normalizer = new ZScoreNormalizer();
            normalizer.FitBatch(data);

            // Klonujemy tablicę, ponieważ TransformInPlace nadpisuje dane wejściowe
            float[] dataToTransform = (float[])data.Clone();

            // ACT
            normalizer.TransformInPlace(dataToTransform);

            // ASSERT
            Assert.Equal(5.0f, normalizer.Mean, precision: 4);
            Assert.Equal(2.0f, normalizer.StandardDeviation, precision: 4);

            // Oczekiwane Z-Scores: z = (x - 5) / 2
            float[] expected = { -1.5f, -0.5f, -0.5f, -0.5f, 0f, 0f, 1.0f, 2.0f };
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], dataToTransform[i], precision: 4);
            }
        }

        [Fact]
        public void TransformInPlace_WithConstantData_ShouldAvoidDivisionByZero()
        {
            // ARRANGE
            // Zbiór danych, gdzie wariancja = 0 (co w naiwnej implementacji powoduje NaN przy dzieleniu)
            float[] data = { 7f, 7f, 7f, 7f, 7f };
            var normalizer = new ZScoreNormalizer();

            normalizer.FitBatch(data);
            float[] dataToTransform = (float[])data.Clone();

            // ACT
            normalizer.TransformInPlace(dataToTransform);

            // ASSERT
            Assert.Equal(7.0f, normalizer.Mean, precision: 4);
            Assert.Equal(0.0f, normalizer.StandardDeviation, precision: 4);

            // Skoro z = (7 - 7) / max(0, 1e-8), wynik musi wynosić 0.0 dla wszystkich elementów
            foreach (var val in dataToTransform)
            {
                Assert.Equal(0f, val, precision: 4);
            }
        }

        [Fact]
        public void Reset_ShouldClearAllInternalStatistics()
        {
            // ARRANGE
            var normalizer = new ZScoreNormalizer();
            normalizer.FitBatch(new float[] { 1f, 2f, 3f }); // Symulacja pracy

            // ACT
            normalizer.Reset();

            // ASSERT
            Assert.Equal(0, normalizer.Count);
            Assert.Equal(0f, normalizer.Mean);
            Assert.Equal(0f, normalizer.StandardDeviation);
        }

        [Fact]
        public void FitBatch_WithEmptySpan_ShouldNotThrowException()
        {
            // ARRANGE
            var normalizer = new ZScoreNormalizer();

            // ACT
            var exception = Record.Exception(() =>
            {
                ReadOnlySpan<float> emptyData = ReadOnlySpan<float>.Empty;
                normalizer.FitBatch(emptyData);
            });

            // ASSERT
            Assert.Null(exception);
            Assert.Equal(0, normalizer.Count);
        }
    }
}