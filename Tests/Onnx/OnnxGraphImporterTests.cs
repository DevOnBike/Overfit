// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;

namespace DevOnBike.Overfit.Tests.Onnx
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
        private const string FixtureDir   = "test_fixtures";
        private const string OnnxFile     = "tiny_resnet.onnx";
        private const string InputBin     = "tiny_resnet_input.bin";
        private const string OutputBin    = "tiny_resnet_output.bin";
        private const int    InputSize    = 8;
        private const int    OutputSize   = 4;
        private const float  Tolerance    = 1e-4f;

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

            Assert.Equal(InputSize,  model.InputSize);
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

            var input  = new float[InputSize];
            var output = new float[OutputSize];

            model.RunInference(input, output);

            Assert.DoesNotContain(output, float.IsNaN);
            Assert.DoesNotContain(output, float.IsInfinity);
        }

        [Fact]
        public void Load_TinyResNet_ViaInferenceEngine_RunsWithoutException()
        {
            SkipIfMissing(OnnxFile);

            using var model   = OnnxGraphImporter.Load(OnnxPath(OnnxFile), InputSize, OutputSize);
            model.Eval();

            var backend = new OnnxGraphInferenceBackend(model);
            using var engine = InferenceEngine.FromBackend(backend);

            var input  = new float[InputSize];
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

            var input    = LoadFloatBin(OnnxPath(InputBin));
            var expected = LoadFloatBin(OnnxPath(OutputBin));
            var output   = new float[OutputSize];

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

            var input    = LoadFloatBin(OnnxPath(InputBin));
            var expected = LoadFloatBin(OnnxPath(OutputBin));
            var output   = new float[OutputSize];

            model.RunInference(input, output);

            var expectedArgmax = Array.IndexOf(expected, expected.Max());
            var actualArgmax   = Array.IndexOf(output,   output.Max());

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

            var input  = new float[InputSize];
            var output = new float[OutputSize];

            // Warmup
            for (var i = 0; i < 256; i++)
            {
                model.RunInference(input, output);
            }

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 10_000; i++)
            {
                model.RunInference(input, output);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0L, allocated);
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
