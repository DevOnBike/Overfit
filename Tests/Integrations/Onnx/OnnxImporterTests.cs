// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
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
            var model = OnnxImporter.Load(
                Path.Combine(
                    FixtureDir,
                    "mnist_cnn.onnx"));

            var paramCount = model.Parameters().Count();

            Assert.True(
                paramCount >= 4,
                $"Expected at least 4 parameter tensors (conv.weight, conv.bias, fc.weight, fc.bias), got {paramCount}.");
        }

        [Fact]
        public void Load_MnistCnn_OutputMatchesPyTorchReference_UsingInferenceEngineRun()
        {
            var model = OnnxImporter.Load(
                Path.Combine(
                    FixtureDir,
                    "mnist_cnn.onnx"));

            model.Eval();

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize: 1 * 28 * 28,
                outputSize: 10);

            var input = LoadFloatBin(
                Path.Combine(
                    FixtureDir,
                    "mnist_input.bin"));

            var expected = LoadFloatBin(
                Path.Combine(
                    FixtureDir,
                    "mnist_output.bin"));

            var output = new float[10];

            engine.Run(
                input,
                output);

            Assert.Equal(
                expected.Length,
                output.Length);

            for (var i = 0; i < output.Length; i++)
            {
                var diff = Math.Abs(output[i] - expected[i]);

                Assert.True(
                    diff <= Tolerance,
                    $"output[{i}] = {output[i]:F6}, expected {expected[i]:F6}, diff = {diff:F6}");
            }

            Assert.Equal(
                Array.IndexOf(expected, expected.Max()),
                Array.IndexOf(output, output.Max()));
        }

        [Fact]
        public void Load_MnistCnn_RunsThroughInferenceEngine_RunPath()
        {
            var model = OnnxImporter.Load(
                Path.Combine(
                    FixtureDir,
                    "mnist_cnn.onnx"));

            model.Eval();

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize: 1 * 28 * 28,
                outputSize: 10);

            var input = new float[784];
            var output = new float[10];

            engine.Run(
                input,
                output);

            Assert.Equal(
                10,
                output.Length);

            Assert.DoesNotContain(
                output,
                float.IsNaN);
        }

        [Fact]
        public void Load_MnistCnn_RunPath_AllocatesZeroBytes()
        {
            var model = OnnxImporter.Load(
                Path.Combine(
                    FixtureDir,
                    "mnist_cnn.onnx"));

            model.Eval();

            using var engine = InferenceEngine.FromSequential(
                model,
                inputSize: 1 * 28 * 28,
                outputSize: 10);

            var input = LoadFloatBin(
                Path.Combine(
                    FixtureDir,
                    "mnist_input.bin"));

            var output = new float[10];

            // Warmup JIT and engine internals outside the measured allocation window.
            for (var i = 0; i < 256; i++)
            {
                engine.Run(
                    input,
                    output);
            }

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 1024; i++)
            {
                engine.Run(
                    input,
                    output);
            }

            var after = GC.GetAllocatedBytesForCurrentThread();
            var allocated = after - before;

            Assert.Equal(
                0,
                allocated);
        }

        private static float[] LoadFloatBin(
            string path)
        {
            var bytes = File.ReadAllBytes(path);

            if (bytes.Length % sizeof(float) != 0)
            {
                throw new InvalidDataException(
                    $"Float fixture length must be divisible by 4: {path}");
            }

            var result = new float[bytes.Length / sizeof(float)];

            for (var i = 0; i < result.Length; i++)
            {
                var bits = BinaryPrimitives.ReadUInt32LittleEndian(
                    bytes.AsSpan(
                        i * sizeof(float),
                        sizeof(float)));

                result[i] = BitConverter.UInt32BitsToSingle(bits);
            }

            return result;
        }
    }
}
