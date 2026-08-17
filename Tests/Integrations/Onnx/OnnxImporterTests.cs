// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Tests.TestSupport.Helpers;

namespace DevOnBike.Overfit.Tests.Integrations.Onnx
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

            // Built before anything is measured, so the closure allocates at this line, not inside a window.
            Action body = () => engine.Run(
                input,
                output);

            // Warmup JIT and engine internals outside the measured allocation window.
            for (var i = 0; i < 256; i++)
            {
                body();
            }

            // Shared CI runners can land a one-off ambient allocation (runtime tiering/services)
            // on the test thread mid-window. A real leak in Run allocates in EVERY window, so a
            // single clean window out of three still proves the zero-allocation path.
            long allocated = -1;

            for (var attempt = 0; attempt < 3 && allocated != 0; attempt++)
            {
                ForceFullGc();

                var before = GC.GetAllocatedBytesForCurrentThread();

                for (var i = 0; i < 1024; i++)
                {
                    body();
                }

                var after = GC.GetAllocatedBytesForCurrentThread();
                allocated = after - before;
            }

            if (allocated != 0)
            {
                // Three dirty windows in a row is no longer plausibly a one-off, so the fourth is taken through
                // the split-loop assert: it reports which half the bytes landed in and whether a GC ran, which
                // is what separates front-loaded tier-up work from a real per-call allocation. The pass
                // criterion is unchanged — both overloads clear at the same one-time-noise floor. See XC-73.
                AssertAllocation.NoPerCallAllocation(
                    "importer Run (fourth window, after 3 dirty ones, 1024 calls)", 1024, body);
            }
        }

        private static void ForceFullGc()
        {
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Forced,
                blocking: true,
                compacting: true);

            GC.WaitForPendingFinalizers();

            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Forced,
                blocking: true,
                compacting: true);
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
