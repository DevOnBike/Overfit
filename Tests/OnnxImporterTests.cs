// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;
using Xunit;

namespace DevOnBike.Overfit.Tests.Onnx
{
    public class OnnxImporterTests
    {
        private const string FixtureDir = "test_fixtures";
        private const float Tolerance = 1e-4f;

        [Fact]
        public void Load_MnistCnn_ReturnsNonEmptySequential()
        {
            // Sequential nie eksponuje listy warstw — weryfikujemy pośrednio:
            // model ma parametry (Conv kernels + bias + FC weights + bias = 4 tensory)
            var model = OnnxImporter.Load(Path.Combine(FixtureDir, "mnist_cnn.onnx"));

            var paramCount = model.Parameters().Count();
            Assert.True(paramCount >= 4,
                $"Expected at least 4 parameter tensors (conv.weight, conv.bias, fc.weight, fc.bias), got {paramCount}");
        }

        [Fact]
        public void Load_MnistCnn_OutputMatchesPyTorchReference()
        {
            var model = OnnxImporter.Load(Path.Combine(FixtureDir, "mnist_cnn.onnx"));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize: 1 * 28 * 28,
                outputSize: 10);

            var input = LoadFloatBin(Path.Combine(FixtureDir, "mnist_input.bin"));
            var expected = LoadFloatBin(Path.Combine(FixtureDir, "mnist_output.bin"));

            var output = engine.Predict(input).ToArray();

            Assert.Equal(expected.Length, output.Length);

            for (var i = 0; i < output.Length; i++)
            {
                Assert.True(
                    Math.Abs(output[i] - expected[i]) <= Tolerance,
                    $"output[{i}] = {output[i]:F6}, expected {expected[i]:F6}, diff = {Math.Abs(output[i] - expected[i]):F6}");
            }

            Assert.Equal(
                Array.IndexOf(expected, expected.Max()),
                Array.IndexOf(output, output.Max()));
        }

        [Fact]
        public void Load_MnistCnn_RunsThroughInferenceEngine_ZeroAllocSmoke()
        {
            var model = OnnxImporter.Load(Path.Combine(FixtureDir, "mnist_cnn.onnx"));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize: 1 * 28 * 28,
                outputSize: 10);

            var output = engine.Predict(new float[784]).ToArray();

            Assert.Equal(10, output.Length);
            Assert.DoesNotContain(output, float.IsNaN);
        }

        private static float[] LoadFloatBin(string path)
        {
            var bytes = File.ReadAllBytes(path);
            return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        }
    }
}