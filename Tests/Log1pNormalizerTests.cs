using DevOnBike.Overfit.Statistical;

namespace DevOnBike.Overfit.Tests
{
    public class Log1pNormalizerTests
    {
        private const int Precision = 4;

        [Fact]
        public void FitBatch_ShouldApplyReluAndLog1pCorrectly()
        {
            // ARRANGE
            // -5f powinno zostać wycięte przez ReLU do 0f. Log1p(0) = 0f.
            // 0f -> Log1p(0) = 0f
            // e-1 (~1.718f) -> Log1p(e-1) = 1f
            float[] data = { -5.0f, 0.0f, (float)Math.E - 1f };
            var normalizer = new Log1pNormalizer();

            // ACT
            normalizer.FitBatch(data);
            normalizer.Freeze();

            // ASSERT
            Assert.Equal(3, normalizer.Count);
            // Oczekiwana średnia z {0, 0, 1} wynosi ~0.3333
            Assert.Equal(1f / 3f, normalizer.Mean, Precision);
        }

        [Fact]
        public void FitBatch_And_FitIncremental_ShouldYieldIdenticalResults()
        {
            // ARRANGE
            float[] data = { -10f, 0f, 5f, 100f, 5000f };
            var batchNormalizer = new Log1pNormalizer();
            var incrementalNormalizer = new Log1pNormalizer();

            // ACT
            batchNormalizer.FitBatch(data);
            batchNormalizer.Freeze();

            foreach (var val in data)
            {
                incrementalNormalizer.FitIncremental(val);
            }
            incrementalNormalizer.Freeze();

            // ASSERT
            Assert.Equal(batchNormalizer.Mean, incrementalNormalizer.Mean, Precision);
            Assert.Equal(batchNormalizer.StdDev, incrementalNormalizer.StdDev, Precision);
        }

        [Fact]
        public void TransformInPlace_WithoutFreeze_ShouldThrowException()
        {
            // ARRANGE
            var normalizer = new Log1pNormalizer();
            normalizer.FitBatch(new float[] { 1f, 2f, 3f });
            var dataToTransform = new float[] { 1f };

            // ACT & ASSERT
            // Zapomnieliśmy wywołać Freeze()
            Assert.Throws<InvalidOperationException>(() => normalizer.TransformInPlace(dataToTransform));
        }

        [Fact]
        public void SaveAndLoad_ShouldPreserveFrozenStatePerfectly()
        {
            // ARRANGE
            var originalNormalizer = new Log1pNormalizer();
            originalNormalizer.FitBatch(new float[] { 10f, 100f, 1000f });
            originalNormalizer.Freeze();

            var loadedNormalizer = new Log1pNormalizer();

            // ACT - Zapis do pamięci operacyjnej i odczyt
            using var memoryStream = new MemoryStream();
            using var writer = new BinaryWriter(memoryStream);
            using var reader = new BinaryReader(memoryStream);

            originalNormalizer.Save(writer);
            
            memoryStream.Position = 0; // Przewijamy taśmę do początku
            loadedNormalizer.Load(reader);

            // ASSERT
            Assert.True(loadedNormalizer.IsFrozen);
            Assert.Equal(originalNormalizer.Mean, loadedNormalizer.Mean, Precision);
            Assert.Equal(originalNormalizer.StdDev, loadedNormalizer.StdDev, Precision);
        }
    }
}