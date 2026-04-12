using System;
using System.Linq;
using Xunit;
using DevOnBike.Overfit.Statistical;

namespace DevOnBike.Overfit.Tests
{
    public class ZScoreNormalizerTests
    {
        // ---------------------------------------------------------------------
        // NOWE TESTY - WERYFIKACJA ALGORYTMU CHANA I MIESZANIA METOD
        // ---------------------------------------------------------------------

        [Fact]
        public void FitBatch_CalledMultipleTimes_ShouldAccumulateCorrectlyUsingChansAlgorithm()
        {
            // ARRANGE
            float[] part1 = [1f, 2f, 3f, 4f, 5f];
            float[] part2 = [10f, 20f, 30f, 40f, 50f];
            float[] part3 = [-5f, -10f, -15f];

            // Scalona tablica jako nasze źródło prawdy (Ground Truth)
            var fullData = part1.Concat(part2).Concat(part3).ToArray();

            var splitNormalizer = new ZScoreNormalizer();
            var fullNormalizer = new ZScoreNormalizer();

            // ACT
            // 1. Fitujemy w 3 osobnych kawałkach (to wymusza zadziałanie Chan's Merge)
            splitNormalizer.FitBatch(part1);
            splitNormalizer.FitBatch(part2);
            splitNormalizer.FitBatch(part3);

            // 2. Fitujemy całość za jednym zamachem
            fullNormalizer.FitBatch(fullData);

            // ASSERT
            Assert.Equal(fullData.Length, splitNormalizer.Count);

            // Średnia i StdDev muszą być identyczne (z dokładnością do 5 miejsc, bo to double pod spodem)
            Assert.Equal(fullNormalizer.Mean, splitNormalizer.Mean, precision: 5);
            Assert.Equal(fullNormalizer.StandardDeviation, splitNormalizer.StandardDeviation, precision: 5);
        }

        [Fact]
        public void FitBatch_And_FitIncremental_CanBeSafelyMixed()
        {
            // ARRANGE
            float[] batch1 = [100f, 200f, 300f];
            var val1 = 150f;
            var val2 = 250f;
            float[] batch2 = [-100f, -200f];

            // Znów tworzymy absolutne źródło prawdy
            float[] fullData = [100f, 200f, 300f, 150f, 250f, -100f, -200f];

            var mixedNormalizer = new ZScoreNormalizer();
            var baselineNormalizer = new ZScoreNormalizer();

            // ACT
            // Mieszamy na pełnej petardzie: Batch -> Incremental -> Incremental -> Batch
            mixedNormalizer.FitBatch(batch1);
            mixedNormalizer.FitIncremental(val1);
            mixedNormalizer.FitIncremental(val2);
            mixedNormalizer.FitBatch(batch2);

            // Baseline fituje po jednym elemencie przez klasycznego Welforda (100% pewności)
            foreach (var val in fullData)
            {
                baselineNormalizer.FitIncremental(val);
            }

            // ASSERT
            Assert.Equal(fullData.Length, mixedNormalizer.Count);
            Assert.Equal(baselineNormalizer.Mean, mixedNormalizer.Mean, precision: 5);
            Assert.Equal(baselineNormalizer.StandardDeviation, mixedNormalizer.StandardDeviation, precision: 5);
        }

        // ---------------------------------------------------------------------
        // ISTNIEJĄCE TESTY (Pozostałe weryfikacje zachowania)
        // ---------------------------------------------------------------------

        [Fact]
        public void FitBatch_And_FitIncremental_ShouldYieldIdenticalResults()
        {
            float[] data = [1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 10.0f, -2.0f, 0.0f];
            var batchNormalizer = new ZScoreNormalizer();
            var incrementalNormalizer = new ZScoreNormalizer();

            batchNormalizer.FitBatch(data);
            foreach (var val in data) incrementalNormalizer.FitIncremental(val);

            Assert.Equal(data.Length, batchNormalizer.Count);
            Assert.Equal(batchNormalizer.Mean, incrementalNormalizer.Mean, precision: 4);
            Assert.Equal(batchNormalizer.StandardDeviation, incrementalNormalizer.StandardDeviation, precision: 4);
        }

        [Fact]
        public void TransformInPlace_ShouldCorrectlyStandardizeData()
        {
            // Średnia = 5.0, StdDev = 2.0
            float[] data = [2f, 4f, 4f, 4f, 5f, 5f, 7f, 9f];
            var normalizer = new ZScoreNormalizer();
            normalizer.FitBatch(data);

            var dataToTransform = (float[])data.Clone();
            normalizer.TransformInPlace(dataToTransform);

            Assert.Equal(5.0f, normalizer.Mean, precision: 4);
            Assert.Equal(2.0f, normalizer.StandardDeviation, precision: 4);

            float[] expected = [-1.5f, -0.5f, -0.5f, -0.5f, 0f, 0f, 1.0f, 2.0f];
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], dataToTransform[i], precision: 4);
            }
        }

        [Fact]
        public void TransformInPlace_WithConstantData_ShouldAvoidDivisionByZero()
        {
            float[] data = [7f, 7f, 7f, 7f, 7f];
            var normalizer = new ZScoreNormalizer();
            normalizer.FitBatch(data);

            var dataToTransform = (float[])data.Clone();
            normalizer.TransformInPlace(dataToTransform);

            Assert.Equal(7.0f, normalizer.Mean, precision: 4);
            Assert.Equal(0.0f, normalizer.StandardDeviation, precision: 4);

            foreach (var val in dataToTransform) Assert.Equal(0f, val, precision: 4);
        }

        [Fact]
        public void Reset_ShouldClearAllInternalStatistics()
        {
            var normalizer = new ZScoreNormalizer();
            normalizer.FitBatch([1f, 2f, 3f]);

            normalizer.Reset();

            Assert.Equal(0, normalizer.Count);
            Assert.Equal(0f, normalizer.Mean);
            Assert.Equal(0f, normalizer.StandardDeviation);
        }

        [Fact]
        public void FitBatch_WithEmptySpan_ShouldNotThrowException()
        {
            var normalizer = new ZScoreNormalizer();

            var exception = Record.Exception(() =>
            {
                var emptyData = ReadOnlySpan<float>.Empty;
                normalizer.FitBatch(emptyData);
            });

            Assert.Null(exception);
            Assert.Equal(0, normalizer.Count);
        }
    }
}