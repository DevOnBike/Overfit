// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit;

namespace DevOnBike.Overfit.Tests
{
    public class TransformerBlockTests
    {
        private const float Tolerance = 1e-3f;

        // ─────────────────────────────────────────────────────────────────────
        // GELU
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GELU_KnownValues_MatchApproximation()
        {
            // GELU(0) = 0; GELU(x) → x for large x; GELU(-x) ≈ 0 for large x
            Assert.True(MathF.Abs(TensorMath.GeluScalar(0f)) < 1e-6f, "GELU(0) should be 0");
            Assert.True(MathF.Abs(TensorMath.GeluScalar(1f) - 0.8413f) < 0.01f, "GELU(1) ≈ 0.841");
            Assert.True(MathF.Abs(TensorMath.GeluScalar(-1f) - (-0.1587f)) < 0.01f, "GELU(-1) ≈ -0.159");
            Assert.True(TensorMath.GeluScalar(5f) > 4.9f, "GELU(large) ≈ x");
        }

        [Fact]
        public void GELU_Backward_NumericalGradientCheck()
        {
            var eps = 1e-3f;
            float[] xs = [-1.5f, -0.5f, 0f, 0.5f, 1.5f];

            foreach (var x in xs)
            {
                var numerical  = (TensorMath.GeluScalar(x + eps) - TensorMath.GeluScalar(x - eps)) / (2 * eps);

                using var g    = new ComputationGraph();
                var storage    = new TensorStorage<float>(1, clearMemory: false);
                storage.AsSpan()[0] = x;
                using var inp  = new AutogradNode(storage, new DevOnBike.Overfit.Tensors.TensorShape(1), requiresGrad: true);
                using var outp = TensorMath.Gelu(g, inp);

                outp.GradView.AsSpan()[0] = 1f;
                g.Backward(outp);

                var analytical = inp.GradView.AsReadOnlySpan()[0];
                var relErr     = MathF.Abs(analytical - numerical) / (MathF.Abs(numerical) + 1e-7f);

                Assert.True(relErr < 0.01f,
                    $"GELU grad at x={x}: analytical={analytical:F6}, numerical={numerical:F6}");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // FeedForwardLayer
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void FFN_OutputShape_IsCorrect()
        {
            using var ffn   = new FeedForwardLayer(dModel: 16, dFF: 64);
            using var graph = new ComputationGraph();
            using var input = MakeInput(2, 5, 16);

            using var output = ffn.Forward(graph, input);

            Assert.Equal(2,  output.Shape.D0);
            Assert.Equal(5,  output.Shape.D1);
            Assert.Equal(16, output.Shape.D2);
        }

        [Fact]
        public void FFN_NoNaNOrInf()
        {
            using var ffn   = new FeedForwardLayer(dModel: 8, dFF: 32);
            using var graph = new ComputationGraph();
            using var input = MakeInput(1, 6, 8);

            using var output = ffn.Forward(graph, input);

            foreach (var v in output.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v),      "NaN in FFN output");
                Assert.False(float.IsInfinity(v), "Inf in FFN output");
            }
        }

        [Fact]
        public void FFN_ParameterCount_IsCorrect()
        {
            // W1[dModel, dFF] + b1[dFF] + W2[dFF, dModel] + b2[dModel]
            const int dModel = 16, dFF = 64;
            using var ffn  = new FeedForwardLayer(dModel, dFF);

            var paramCount = ffn.TrainableParameters().Sum(p => p.Shape.Size);
            var expected   = dModel * dFF + dFF + dFF * dModel + dModel;

            Assert.Equal(expected, paramCount);
        }

        // ─────────────────────────────────────────────────────────────────────
        // TransformerBlock
        // ─────────────────────────────────────────────────────────────────────

        [Theory]
        [InlineData(true)]   // Pre-LN (modern)
        [InlineData(false)]  // Post-LN (original GPT-1)
        public void TransformerBlock_OutputShape_MatchesInput(bool preLN)
        {
            const int dModel = 16, nHeads = 4, dFF = 64, batch = 2, seq = 5;

            using var block  = new TransformerBlock(dModel, nHeads, dFF, preLayerNorm: preLN);
            using var graph  = new ComputationGraph();
            using var input  = MakeInput(batch, seq, dModel);

            using var output = block.Forward(graph, input);

            Assert.Equal(batch,  output.Shape.D0);
            Assert.Equal(seq,    output.Shape.D1);
            Assert.Equal(dModel, output.Shape.D2);
        }

        [Fact]
        public void TransformerBlock_NoNaNOrInf()
        {
            using var block  = new TransformerBlock(16, 4, 64, causalMask: true, preLayerNorm: true);
            using var graph  = new ComputationGraph();
            using var input  = MakeInput(1, 8, 16);

            using var output = block.Forward(graph, input);

            foreach (var v in output.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v),      "NaN in TransformerBlock output");
                Assert.False(float.IsInfinity(v), "Inf in TransformerBlock output");
            }
        }

        [Fact]
        public void TransformerBlock_ParameterCount_IsCorrect()
        {
            // LN1: 2*dModel
            // MHA: 4*dModel² + dModel
            // LN2: 2*dModel
            // FFN: 2*dModel*dFF + dFF + dModel
            const int dModel = 16, nHeads = 4, dFF = 64;
            using var block = new TransformerBlock(dModel, nHeads, dFF);

            var paramCount = block.TrainableParameters().Sum(p => p.Shape.Size);
            var expected   = 2 * dModel                          // LN1
                           + 4 * dModel * dModel + dModel        // MHA
                           + 2 * dModel                          // LN2
                           + 2 * dModel * dFF + dFF + dModel;    // FFN

            Assert.Equal(expected, paramCount);
        }

        [Fact]
        public void TransformerBlock_TrainEval_DoesNotThrow()
        {
            using var block = new TransformerBlock(16, 4, 64);
            block.Train();
            block.Eval();
            block.Train();
        }

        [Fact]
        public void TransformerBlock_SaveLoad_ProducesSameOutput()
        {
            const int dModel = 8, nHeads = 2, dFF = 32, batch = 1, seq = 3;

            using var block1 = new TransformerBlock(dModel, nHeads, dFF);
            using var ms     = new System.IO.MemoryStream();
            using var bw     = new System.IO.BinaryWriter(ms);
            block1.Save(bw);

            ms.Position = 0;
            using var br     = new System.IO.BinaryReader(ms);
            using var block2 = new TransformerBlock(dModel, nHeads, dFF);
            block2.Load(br);

            var inputData = MakeRandom(batch * seq * dModel, seed: 7);

            using var g1  = new ComputationGraph();
            using var i1  = MakeInputFromData(inputData, batch, seq, dModel);
            using var o1  = block1.Forward(g1, i1);

            using var g2  = new ComputationGraph();
            using var i2  = MakeInputFromData(inputData, batch, seq, dModel);
            using var o2  = block2.Forward(g2, i2);

            var s1 = o1.DataView.AsReadOnlySpan();
            var s2 = o2.DataView.AsReadOnlySpan();

            for (var i = 0; i < s1.Length; i++)
            {
                Assert.True(MathF.Abs(s1[i] - s2[i]) < Tolerance,
                    $"[{i}] after load: {s1[i]:F6} vs {s2[i]:F6}");
            }
        }

        [Fact]
        public void TransformerBlock_GPT1Scale_ForwardRunsWithoutException()
        {
            // GPT-1: 12 blocks, dModel=768, nHeads=12, dFF=3072
            // Just smoke-test 1 block with small seq
            const int dModel = 768, nHeads = 12, dFF = 3072, batch = 1, seq = 4;

            using var block  = new TransformerBlock(dModel, nHeads, dFF, causalMask: true);
            using var graph  = new ComputationGraph();
            using var input  = MakeInput(batch, seq, dModel);

            using var output = block.Forward(graph, input);

            Assert.Equal(batch,  output.Shape.D0);
            Assert.Equal(seq,    output.Shape.D1);
            Assert.Equal(dModel, output.Shape.D2);

            foreach (var v in output.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v),      "NaN in GPT-1 scale block output");
                Assert.False(float.IsInfinity(v), "Inf in GPT-1 scale block output");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helpers
        // ─────────────────────────────────────────────────────────────────────

        private static AutogradNode MakeInput(int b, int t, int d)
        {
            return MakeInputFromData(MakeRandom(b * t * d, seed: 42), b, t, d);
        }

        private static AutogradNode MakeInputFromData(float[] data, int b, int t, int d)
        {
            var storage = new TensorStorage<float>(data.Length, clearMemory: false);
            data.AsSpan().CopyTo(storage.AsSpan());
            return new AutogradNode(storage, new TensorShape(b, t, d), requiresGrad: true);
        }

        private static float[] MakeRandom(int size, int seed)
        {
            var rng  = new Random(seed);
            var data = new float[size];
            for (var i = 0; i < size; i++)
            {
                data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
            }
            return data;
        }
    }
}
