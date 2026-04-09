// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class AnomalyAutoencoderTests
    {
        // -------------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------------

        private const int DefaultInputSize = 32; // 8 cech × 4 statystyki
        private const float Tolerance = 1e-5f;

        private static AnomalyAutoencoder Make(int inputSize = DefaultInputSize)
            => new(inputSize);

        private static AnomalyAutoencoder MakeEval(int inputSize = DefaultInputSize)
        {
            var m = new AnomalyAutoencoder(inputSize);
            m.Eval();
            return m;
        }

        private static float[] Features(int size, float value = 0.5f)
        {
            var f = new float[size];
            f.AsSpan().Fill(value);
            return f;
        }

        private static float[] Reconstruction(int size) => new float[size];

        // -------------------------------------------------------------------------
        // Constructor
        // -------------------------------------------------------------------------

        [Fact]
        public void Constructor_WhenInputSizeIsZero_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(() => new AnomalyAutoencoder(0));

        [Fact]
        public void Constructor_WhenInputSizeIsNegative_ThenThrowsArgumentOutOfRange()
            => Assert.Throws<ArgumentOutOfRangeException>(() => new AnomalyAutoencoder(-1));

        [Fact]
        public void Constructor_WhenCreated_ThenIsTrainingIsTrue()
        {
            using var m = Make();
            Assert.True(m.IsTraining);
        }

        [Fact]
        public void Constructor_WhenCreated_ThenInputSizeMatchesArg()
        {
            using var m = Make(inputSize: 16);
            Assert.Equal(16, m.InputSize);
        }

        [Theory]
        [InlineData(32, 0, 0, 0, 4)] // domyślne: 32/8=4
        [InlineData(16, 0, 0, 0, 2)] // 16/8=2
        [InlineData(32, 0, 0, 8, 8)] // jawny bottleneck
        public void Constructor_WhenBottleneckDimIsDefault_ThenEqualsInputSizeDiv8(
            int inputSize, int h1, int h2, int bn, int expected)
        {
            using var m = new AnomalyAutoencoder(inputSize, h1, h2, bn);
            Assert.Equal(expected, m.BottleneckDim);
        }

        // -------------------------------------------------------------------------
        // Train / Eval
        // -------------------------------------------------------------------------

        [Fact]
        public void Eval_WhenCalled_ThenIsTrainingIsFalse()
        {
            using var m = Make();
            m.Eval();
            Assert.False(m.IsTraining);
        }

        [Fact]
        public void Train_WhenCalledAfterEval_ThenIsTrainingIsTrue()
        {
            using var m = Make();
            m.Eval();
            m.Train();
            Assert.True(m.IsTraining);
        }

        // -------------------------------------------------------------------------
        // Reconstruct — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Reconstruct_WhenFeaturesLengthMismatch_ThenThrowsArgumentException()
        {
            using var m = MakeEval();
            var features = new float[DefaultInputSize - 1]; // za krótkie
            var output = Reconstruction(DefaultInputSize);

            Assert.Throws<ArgumentException>(() => m.Reconstruct(features, output));
        }

        [Fact]
        public void Reconstruct_WhenOutputBufferTooShort_ThenThrowsArgumentException()
        {
            using var m = MakeEval();
            var features = Features(DefaultInputSize);
            var output = new float[DefaultInputSize - 1]; // za krótki

            Assert.Throws<ArgumentException>(() => m.Reconstruct(features, output));
        }

        // -------------------------------------------------------------------------
        // Reconstruct — poprawność wyjścia
        // -------------------------------------------------------------------------

        [Fact]
        public void Reconstruct_WhenCalledInEvalMode_ThenOutputIsFinite()
        {
            using var m = MakeEval();
            var features = Features(DefaultInputSize, value: 0.3f);
            var output = Reconstruction(DefaultInputSize);

            m.Reconstruct(features, output);

            foreach (var v in output)
            {
                Assert.True(float.IsFinite(v), $"Non-finite value in reconstruction: {v}");
            }
        }

        [Fact]
        public void Reconstruct_WhenCalledTwiceWithSameInput_ThenOutputIsDeterministic()
        {
            using var m = MakeEval();
            var features = Features(DefaultInputSize, value: 0.7f);
            var out1 = Reconstruction(DefaultInputSize);
            var out2 = Reconstruction(DefaultInputSize);

            m.Reconstruct(features, out1);
            m.Reconstruct(features, out2);

            for (var i = 0; i < DefaultInputSize; i++)
            {
                Assert.Equal(out1[i], out2[i]);
            }
        }

        [Fact]
        public void Reconstruct_WhenCalledWithAllZeros_ThenOutputIsFinite()
        {
            using var m = MakeEval();
            var features = new float[DefaultInputSize]; // wszystkie zera
            var output = Reconstruction(DefaultInputSize);

            m.Reconstruct(features, output); // nie rzuca, wynik skończony

            foreach (var v in output)
            {
                Assert.True(float.IsFinite(v), $"Non-finite value: {v}");
            }
        }

        [Fact]
        public void Reconstruct_WhenCalledWithDifferentInputs_ThenOutputsDiffer()
        {
            using var m = MakeEval();

            var rng = new Random(42);
            
            foreach (var p in m.Parameters())
            {
                var span = p.Data.AsSpan();
                for (var i = 0; i < span.Length; i++)
                {
                    span[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
                }
            }

            var normal = Features(DefaultInputSize, value: 0.2f);
            var anomaly = Features(DefaultInputSize, value: 5.0f);
            var outNormal = Reconstruction(DefaultInputSize);
            var outAnomaly = Reconstruction(DefaultInputSize);

            m.Reconstruct(normal, outNormal);
            m.Reconstruct(anomaly, outAnomaly);

            // Rekonstrukcje muszą się różnić
            Assert.False(outNormal.SequenceEqual(outAnomaly));
        }

        [Fact]
        public void Reconstruct_WhenOutputBufferLargerThanNeeded_ThenDoesNotThrow()
        {
            using var m = MakeEval();
            var features = Features(DefaultInputSize);
            var output = new float[DefaultInputSize + 16]; // większy niż potrzeba

            m.Reconstruct(features, output); // nie rzuca
        }

        // -------------------------------------------------------------------------
        // Reconstruct — nie mutuje wejścia
        // -------------------------------------------------------------------------

        [Fact]
        public void Reconstruct_WhenCalled_ThenInputFeaturesAreNotMutated()
        {
            using var m = MakeEval();
            var features = Features(DefaultInputSize, value: 0.42f);
            var original = features.ToArray();
            var output = Reconstruction(DefaultInputSize);

            m.Reconstruct(features, output);

            Assert.Equal(original, features);
        }

        // -------------------------------------------------------------------------
        // Forward — training path
        // -------------------------------------------------------------------------

        [Fact]
        public void Forward_WhenCalledWithNullGraph_ThenReturnsAutogradNode()
        {
            using var m = MakeEval();
            using var input = new AutogradNode(new FastTensor<float>(1, DefaultInputSize), false);
            Features(DefaultInputSize).CopyTo(input.Data.AsSpan());

            var result = m.Forward(null, input);

            Assert.NotNull(result);
            Assert.Equal(DefaultInputSize, result.Data.Size);
        }

        [Fact]
        public void Forward_WhenCalledWithGraph_ThenReturnsAutogradNode()
        {
            using var m = new AnomalyAutoencoder(DefaultInputSize);
            m.Train();
            var graph = new ComputationGraph();
            using var input = new AutogradNode(new FastTensor<float>(1, DefaultInputSize), false);
            Features(DefaultInputSize).CopyTo(input.Data.AsSpan());

            var result = m.Forward(graph, input);

            Assert.NotNull(result);
            Assert.Equal(DefaultInputSize, result.Data.Size);
        }

        // -------------------------------------------------------------------------
        // Parameters
        // -------------------------------------------------------------------------

        [Fact]
        public void Parameters_WhenCalled_ThenReturnsNonEmpty()
        {
            using var m = Make();
            Assert.NotEmpty(m.Parameters());
        }

        [Fact]
        public void Parameters_WhenCalled_ThenCountIsPositive()
        {
            using var m = Make();
            Assert.True(m.ParameterCount > 0);
        }

        [Theory]
        [InlineData(32, 16, 8, 4)] // domyślna konfiguracja
        [InlineData(16, 8, 4, 2)] // mniejsza sieć
        public void Parameters_WhenArchitectureDefined_ThenCountMatchesExpected(
            int inputSize, int h1, int h2, int bn)
        {
            using var m = new AnomalyAutoencoder(inputSize, h1, h2, bn);

            // Encoder: 3 × LinearLayer (weights + biases) + 3 × BatchNorm1D (gamma + beta)
            // Decoder: 3 × LinearLayer (weights + biases) + 2 × BatchNorm1D (gamma + beta)
            // Encoder Linear params: inputSize*h1 + h1 + h1*h2 + h2 + h2*bn + bn
            // Encoder BN params: h1*2 + h2*2 + bn*2
            // Decoder Linear params: bn*h2 + h2 + h2*h1 + h1 + h1*inputSize + inputSize
            // Decoder BN params: h2*2 + h1*2
            var encLinear = inputSize * h1 + h1 + h1 * h2 + h2 + h2 * bn + bn;
            var encBn = h1 * 2 + h2 * 2 + bn * 2;
            var decLinear = bn * h2 + h2 + h2 * h1 + h1 + h1 * inputSize + inputSize;
            var decBn = h2 * 2 + h1 * 2;
            var expected = encLinear + encBn + decLinear + decBn;

            Assert.Equal(expected, m.ParameterCount);
        }

        // -------------------------------------------------------------------------
        // Save / Load — round-trip
        // -------------------------------------------------------------------------

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenReconstructionIsIdentical()
        {
            using var m1 = MakeEval();
            using var m2 = new AnomalyAutoencoder(DefaultInputSize);
            m2.Eval();

            var features = Features(DefaultInputSize, value: 0.33f);
            var out1 = Reconstruction(DefaultInputSize);
            var out2 = Reconstruction(DefaultInputSize);

            // Nagraj wyjście przed save
            m1.Reconstruct(features, out1);

            // Save → Load do m2
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            m1.Save(bw);
            bw.Flush();

            ms.Position = 0;
            using var br = new BinaryReader(ms);
            m2.Load(br);

            // Rekonstrukcja po load musi być identyczna
            m2.Reconstruct(features, out2);

            for (var i = 0; i < DefaultInputSize; i++)
            {
                Assert.Equal(out1[i], out2[i]);
            }
        }

        [Fact]
        public void Save_WhenPathProvided_ThenFileIsCreated()
        {
            using var m = Make();
            var path = Path.GetTempFileName();

            try
            {
                m.Save(path);
                Assert.True(File.Exists(path));
                Assert.True(new FileInfo(path).Length > 0);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_WhenFileDoesNotExist_ThenThrowsFileNotFoundException()
        {
            using var m = Make();
            Assert.Throws<FileNotFoundException>(() => m.Load("/tmp/nieistniejacy_model.bin"));
        }

        // -------------------------------------------------------------------------
        // Dispose
        // -------------------------------------------------------------------------

        [Fact]
        public void Dispose_WhenCalled_ThenDoesNotThrow()
        {
            var m = Make();
            m.Dispose(); // nie rzuca
        }

        [Fact]
        public void Dispose_WhenCalledTwice_ThenIsIdempotent()
        {
            var m = Make();
            m.Dispose();
            m.Dispose(); // nie rzuca
        }
    }
}