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
    /// <summary>
    /// Integration tests for <see cref="OnnxGraphImporter"/> — the DAG-topology ONNX importer.
    ///
    /// Test model: TinyResNet
    ///   fc1 = Linear(8, 8)
    ///   fc2 = Linear(8, 4)
    ///   forward(x) = fc2(relu(fc1(x)) + x)   ← residual / skip connection
    ///
    /// Fixture files (generate with fixture.py):
    ///   Tests/test_fixtures/tiny_resnet.onnx
    ///   Tests/test_fixtures/tiny_resnet_input.bin    (8 float32 LE)
    ///   Tests/test_fixtures/tiny_resnet_output.bin   (4 float32 LE)
    ///
    /// Tests skip gracefully when fixtures are absent so CI without Python
    /// does not block the build.
    /// </summary>
    public class OnnxGraphImporterTests
    {
        private const string FixtureDir = "test_fixtures";
        private const string OnnxFile = "tiny_resnet.onnx";
        private const string InputBin = "tiny_resnet_input.bin";
        private const string OutputBin = "tiny_resnet_output.bin";
        private const int InputSize = 8;
        private const int OutputSize = 4;
        private const float Tolerance = 1e-4f;

        // ─────────────────────────────────────────────────────────────────────
        // Structural tests (no reference I/O needed)
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Load_TinyResNet_ReturnsModel()
        {
            SkipIfMissing(OnnxFile);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(OnnxFile),
                InputSize,
                OutputSize);

            Assert.Equal(InputSize, model.InputSize);
            Assert.Equal(OutputSize, model.OutputSize);
        }

        [Fact]
        public void Load_TinyResNet_RunsForward_WithoutException()
        {
            SkipIfMissing(OnnxFile);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(OnnxFile),
                InputSize,
                OutputSize);

            model.Eval();

            var input = new float[InputSize];
            var output = new float[OutputSize];

            model.RunInference(input, output);

            Assert.DoesNotContain(output, float.IsNaN);
            Assert.DoesNotContain(output, float.IsInfinity);
        }

        [Fact]
        public void Load_TinyResNet_ViaInferenceEngine_RunsWithoutException()
        {
            SkipIfMissing(OnnxFile);

            using var model = OnnxGraphImporter.Load(OnnxPath(OnnxFile), InputSize, OutputSize);
            model.Eval();

            var backend = new OnnxGraphInferenceBackend(model);
            using var engine = InferenceEngine.FromBackend(backend);

            var input = new float[InputSize];
            var output = engine.Predict(input);

            Assert.Equal(OutputSize, output.Length);
            Assert.DoesNotContain(output.ToArray(), float.IsNaN);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Numerical parity with PyTorch reference
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Load_TinyResNet_OutputMatchesPyTorchReference()
        {
            SkipIfMissing(OnnxFile, InputBin, OutputBin);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(OnnxFile),
                InputSize,
                OutputSize);

            model.Eval();

            var input = LoadFloatBin(OnnxPath(InputBin));
            var expected = LoadFloatBin(OnnxPath(OutputBin));
            var output = new float[OutputSize];

            model.RunInference(input, output);

            Assert.Equal(expected.Length, output.Length);

            for (var i = 0; i < output.Length; i++)
            {
                var diff = MathF.Abs(output[i] - expected[i]);
                Assert.True(
                    diff <= Tolerance,
                    $"output[{i}] = {output[i]:F6}, expected {expected[i]:F6}, diff = {diff:F6}");
            }
        }

        [Fact]
        public void Load_TinyResNet_ArgmaxMatchesPyTorch()
        {
            SkipIfMissing(OnnxFile, InputBin, OutputBin);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(OnnxFile),
                InputSize,
                OutputSize);

            model.Eval();

            var input = LoadFloatBin(OnnxPath(InputBin));
            var expected = LoadFloatBin(OnnxPath(OutputBin));
            var output = new float[OutputSize];

            model.RunInference(input, output);

            var expectedArgmax = Array.IndexOf(expected, expected.Max());
            var actualArgmax = Array.IndexOf(output, output.Max());

            Assert.Equal(expectedArgmax, actualArgmax);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Zero-allocation inference
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Load_TinyResNet_InferenceAllocatesZeroBytes()
        {
            SkipIfMissing(OnnxFile);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(OnnxFile),
                InputSize,
                OutputSize);

            model.Eval();

            var input = new float[InputSize];
            var output = new float[OutputSize];

            AssertZeroAllocPerInference(() => model.RunInference(input, output), "tiny_resnet inference");
        }

        // Asserts ZERO per-inference allocation. A genuine per-call leak allocates a whole object (>= ~24 B) every
        // iteration, scaling to hundreds of KB over the loop; some runtimes instead charge a single one-time JIT
        // tier-up / PGO / OSR bookkeeping allocation (observed ~280 B) to this thread inside the measured window —
        // it lands on whichever inference test the tier-up happens during, hence the cross-platform flakiness. That
        // is runtime infrastructure, not the inference path, so tolerate a strict sub-1-byte-per-call floor.
        private static void AssertZeroAllocPerInference(Action runInference, string label)
        {
            for (var i = 0; i < 256; i++)
            {
                runInference();
            }

            const int iterations = 10_000;
            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < iterations; i++)
            {
                runInference();
            }
            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            AllocationAssert.NoPerCallAllocation(allocated, label);
        }

        // ─────────────────────────────────────────────────────────────────────
        // AveragePool model: Conv→ReLU→AveragePool→GAP→Linear
        // Loaded via OnnxImporter (linear topology, no skip connections)
        // ─────────────────────────────────────────────────────────────────────

        private const string AvgPoolOnnx = "avgpool_model.onnx";
        private const string AvgPoolInput = "avgpool_model_input.bin";
        private const string AvgPoolOutput = "avgpool_model_output.bin";
        private const int AvgPoolIn = 64;  // 1 * 1 * 8 * 8
        private const int AvgPoolOut = 2;

        [Fact]
        public void Load_AvgPoolModel_Sequential_LoadsWithoutException()
        {
            SkipIfMissing(AvgPoolOnnx);

            var model = OnnxImporter.Load(OnnxPath(AvgPoolOnnx));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(model, AvgPoolIn, AvgPoolOut);

            var input = new float[AvgPoolIn];
            var output = new float[AvgPoolOut];

            engine.Run(input, output);

            Assert.DoesNotContain(output, float.IsNaN);
        }

        [Fact]
        public void Load_AvgPoolModel_OutputMatchesPyTorchReference()
        {
            SkipIfMissing(AvgPoolOnnx, AvgPoolInput, AvgPoolOutput);

            var model = OnnxImporter.Load(OnnxPath(AvgPoolOnnx));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(model, AvgPoolIn, AvgPoolOut);

            var input = LoadFloatBin(OnnxPath(AvgPoolInput));
            var expected = LoadFloatBin(OnnxPath(AvgPoolOutput));
            var output = new float[AvgPoolOut];

            engine.Run(input, output);

            for (var i = 0; i < output.Length; i++)
            {
                var diff = MathF.Abs(output[i] - expected[i]);
                Assert.True(
                    diff <= Tolerance,
                    $"avgpool output[{i}] = {output[i]:F6}, expected {expected[i]:F6}, diff = {diff:F6}");
            }
        }

        [Fact]
        public void Load_AvgPoolModel_AllocatesZeroBytes()
        {
            SkipIfMissing(AvgPoolOnnx);

            var model = OnnxImporter.Load(OnnxPath(AvgPoolOnnx));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(model, AvgPoolIn, AvgPoolOut);

            var input = new float[AvgPoolIn];
            var output = new float[AvgPoolOut];

            AssertZeroAllocPerInference(() => engine.Run(input, output), "avgpool inference");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helpers
        // ─────────────────────────────────────────────────────────────────────

        private static string OnnxPath(string file) => Path.Combine(FixtureDir, file);

        private static void SkipIfMissing(params string[] files)
        {
            foreach (var file in files)
            {
                if (!File.Exists(OnnxPath(file)))
                {
                    throw new Exception(
                        $"Fixture '{file}' not found in '{FixtureDir}'. " +
                        "Run fixture.py to generate test fixtures.");
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // ResNetBlock: Conv2d(padding=1) + skip connection + BatchNorm (folded)
        // ─────────────────────────────────────────────────────────────────────

        private const string ResNetOnnx = "resnet_block.onnx";
        private const string ResNetInput = "resnet_block_input.bin";
        private const string ResNetOutput = "resnet_block_output.bin";
        private const int ResNetFlat = 256; // 1 * 4 * 8 * 8

        [Fact]
        public void Load_ResNetBlock_ReturnsModel()
        {
            SkipIfMissing(ResNetOnnx);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(ResNetOnnx),
                ResNetFlat, ResNetFlat);

            Assert.Equal(ResNetFlat, model.InputSize);
            Assert.Equal(ResNetFlat, model.OutputSize);
        }

        [Fact]
        public void Load_ResNetBlock_OutputMatchesPyTorchReference()
        {
            SkipIfMissing(ResNetOnnx, ResNetInput, ResNetOutput);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(ResNetOnnx),
                ResNetFlat, ResNetFlat);

            model.Eval();

            var input = LoadFloatBin(OnnxPath(ResNetInput));
            var expected = LoadFloatBin(OnnxPath(ResNetOutput));
            var output = new float[ResNetFlat];

            model.RunInference(input, output);

            for (var i = 0; i < output.Length; i++)
            {
                var diff = MathF.Abs(output[i] - expected[i]);
                Assert.True(
                    diff <= Tolerance,
                    $"resnet_block output[{i}] = {output[i]:F6}, " +
                    $"expected {expected[i]:F6}, diff = {diff:F6}");
            }
        }

        [Fact]
        public void Load_ResNetBlock_InferenceAllocatesZeroBytes()
        {
            SkipIfMissing(ResNetOnnx);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(ResNetOnnx),
                ResNetFlat, ResNetFlat);

            model.Eval();

            var input = new float[ResNetFlat];
            var output = new float[ResNetFlat];

            AssertZeroAllocPerInference(() => model.RunInference(input, output), "resnet_block inference");
        }

        private static float[] LoadFloatBin(string path)
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
                result[i] = BitConverter.UInt32BitsToSingle(
                    BinaryPrimitives.ReadUInt32LittleEndian(
                        bytes.AsSpan(i * sizeof(float), sizeof(float))));
            }

            return result;
        }
    }
}
