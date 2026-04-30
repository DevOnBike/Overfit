// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;
using Xunit;

namespace DevOnBike.Overfit.Tests.Onnx
{
    /// <summary>
    /// Integration tests for:
    ///   - <see cref="AveragePool2DLayer"/> via ONNX import (tiny_avgpool.onnx)
    ///   - <see cref="OnnxGraphImporter"/> with ResNetBlock (resnet_block.onnx)
    ///
    /// Generate fixtures:
    ///   python tests/fixture_avgpool.py
    ///
    /// All tests skip gracefully when fixture files are absent.
    /// </summary>
    public class AveragePoolAndDagTests
    {
        private const string FixtureDir = "test_fixtures";
        private const float  Tolerance  = 1e-4f;

        // ─────────────────────────────────────────────────────────────────────
        // AveragePool via OnnxImporter (linear topology)
        // ─────────────────────────────────────────────────────────────────────

        private const string AvgPoolOnnx   = "tiny_avgpool.onnx";
        private const string AvgPoolInput  = "tiny_avgpool_input.bin";
        private const string AvgPoolOutput = "tiny_avgpool_output.bin";
        private const int    AvgPoolIn     = 256;  // 1×4×8×8
        private const int    AvgPoolOut    = 10;

        [Fact]
        public void AveragePool_Load_ReturnsNonEmptySequential()
        {
            SkipIfMissing(AvgPoolOnnx);

            var model = OnnxImporter.Load(OnnxPath(AvgPoolOnnx));

            Assert.True(
                model.Parameters().Any(),
                "Model should have at least one parameter (conv + fc weights).");
        }

        [Fact]
        public void AveragePool_OutputMatchesPyTorchReference()
        {
            SkipIfMissing(AvgPoolOnnx, AvgPoolInput, AvgPoolOutput);

            var model = OnnxImporter.Load(OnnxPath(AvgPoolOnnx));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(model, AvgPoolIn, AvgPoolOut);

            var input    = LoadFloatBin(OnnxPath(AvgPoolInput));
            var expected = LoadFloatBin(OnnxPath(AvgPoolOutput));
            var output   = new float[AvgPoolOut];

            engine.Run(input, output);

            for (var i = 0; i < AvgPoolOut; i++)
            {
                var diff = MathF.Abs(output[i] - expected[i]);
                Assert.True(
                    diff <= Tolerance,
                    $"output[{i}] = {output[i]:F6}, expected {expected[i]:F6}, diff = {diff:F6}");
            }
        }

        [Fact]
        public void AveragePool_InferenceAllocatesZeroBytes()
        {
            SkipIfMissing(AvgPoolOnnx);

            var model = OnnxImporter.Load(OnnxPath(AvgPoolOnnx));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(model, AvgPoolIn, AvgPoolOut);

            var input  = new float[AvgPoolIn];
            var output = new float[AvgPoolOut];

            for (var i = 0; i < 256; i++) engine.Run(input, output);

            var before    = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 10_000; i++) engine.Run(input, output);
            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0L, allocated);
        }

        [Fact]
        public void AveragePool_ArgmaxMatchesPyTorch()
        {
            SkipIfMissing(AvgPoolOnnx, AvgPoolInput, AvgPoolOutput);

            var model = OnnxImporter.Load(OnnxPath(AvgPoolOnnx));
            model.Eval();

            using var engine = InferenceEngine.FromSequential(model, AvgPoolIn, AvgPoolOut);

            var input    = LoadFloatBin(OnnxPath(AvgPoolInput));
            var expected = LoadFloatBin(OnnxPath(AvgPoolOutput));
            var output   = new float[AvgPoolOut];

            engine.Run(input, output);

            Assert.Equal(
                Array.IndexOf(expected, expected.Max()),
                Array.IndexOf(output,   output.Max()));
        }

        // ─────────────────────────────────────────────────────────────────────
        // AveragePool2DLayer unit tests (no ONNX — pure math)
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void AveragePool2DLayer_ForwardInference_2x2_NoStride_NopadCorrect()
        {
            // Input: [1, 1, 4, 4] (1 batch, 1 channel, 4×4)
            // Kernel: 2×2, stride=2, padding=0  → output [1, 1, 2, 2]
            var layer = new AveragePool2DLayer(
                channels: 1, inputH: 4, inputW: 4,
                kernelSize: 2, padding: 0, stride: 2);

            // Input:
            //  1 2 3 4
            //  5 6 7 8
            //  9 0 1 2
            //  3 4 5 6
            var input = new float[]
            {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 0, 1, 2,
                3, 4, 5, 6,
            };

            // Expected averages:
            //  top-left:  (1+2+5+6)/4 = 3.5
            //  top-right: (3+4+7+8)/4 = 5.5
            //  bot-left:  (9+0+3+4)/4 = 4.0
            //  bot-right: (1+2+5+6)/4 = 3.5
            var output   = new float[4];
            layer.ForwardInference(input, output);

            Assert.Equal(3.5f, output[0], precision: 5);
            Assert.Equal(5.5f, output[1], precision: 5);
            Assert.Equal(4.0f, output[2], precision: 5);
            Assert.Equal(3.5f, output[3], precision: 5);
        }

        [Fact]
        public void AveragePool2DLayer_ForwardInference_Padding1_CorrectOutputSize()
        {
            // Input: [1, 1, 4, 4], kernel=3, padding=1, stride=1
            // outH = (4 + 2 - 3) / 1 + 1 = 4 → same spatial size
            var layer = new AveragePool2DLayer(
                channels: 1, inputH: 4, inputW: 4,
                kernelSize: 3, padding: 1, stride: 1);

            Assert.Equal(16, layer.InferenceOutputSize); // 1 * 4 * 4
            Assert.Equal(16, layer.InferenceInputSize);

            var input  = Enumerable.Range(1, 16).Select(x => (float)x).ToArray();
            var output = new float[16];

            // Should not throw and should fill output without NaN
            layer.ForwardInference(input, output);
            Assert.DoesNotContain(output, float.IsNaN);
        }

        // ─────────────────────────────────────────────────────────────────────
        // ResNetBlock — DAG topology (Add operator = skip connection)
        // ─────────────────────────────────────────────────────────────────────

        private const string ResNetOnnx   = "resnet_block.onnx";
        private const string ResNetInput  = "resnet_block_input.bin";
        private const string ResNetOutput = "resnet_block_output.bin";
        private const int    ResNetFlat   = 256;  // 1×4×8×8

        [Fact]
        public void ResNetBlock_DAG_Load_Succeeds()
        {
            SkipIfMissing(ResNetOnnx);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(ResNetOnnx), ResNetFlat, ResNetFlat);

            Assert.Equal(ResNetFlat, model.InputSize);
            Assert.Equal(ResNetFlat, model.OutputSize);
        }

        [Fact]
        public void ResNetBlock_DAG_OutputMatchesPyTorchReference()
        {
            SkipIfMissing(ResNetOnnx, ResNetInput, ResNetOutput);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(ResNetOnnx), ResNetFlat, ResNetFlat);

            model.Eval();

            var input    = LoadFloatBin(OnnxPath(ResNetInput));
            var expected = LoadFloatBin(OnnxPath(ResNetOutput));
            var output   = new float[ResNetFlat];

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
        public void ResNetBlock_DAG_InferenceAllocatesZeroBytes()
        {
            SkipIfMissing(ResNetOnnx);

            using var model = OnnxGraphImporter.Load(
                OnnxPath(ResNetOnnx), ResNetFlat, ResNetFlat);

            model.Eval();

            var input  = new float[ResNetFlat];
            var output = new float[ResNetFlat];

            for (var i = 0; i < 256; i++) model.RunInference(input, output);

            var before    = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 10_000; i++) model.RunInference(input, output);
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
                        $"Fixture '{file}' not found. Run tests/fixture_avgpool.py.");
                }
            }
        }

        private static float[] LoadFloatBin(string path)
        {
            var bytes  = File.ReadAllBytes(path);
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
