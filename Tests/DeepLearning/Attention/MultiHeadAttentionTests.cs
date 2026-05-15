// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.DeepLearning.Attention
{
    /// <summary>
    /// Tests for MultiHeadAttentionLayer.
    /// </summary>
    public class MultiHeadAttentionTests
    {
        private const float Tolerance = 1e-3f;

        [Theory]
        [InlineData(8, 2, 2, 4)]
        [InlineData(16, 4, 3, 6)]
        [InlineData(32, 8, 2, 5)]
        public void MHA_OutputShape_MatchesInputShape(int dModel, int nHeads, int batch, int seq)
        {
            using var mha = new MultiHeadAttentionLayer(dModel, nHeads, causalMask: true);
            using var graph = new ComputationGraph();
            using var input = MakeInput(batch, seq, dModel, requiresGrad: false);
            using var output = mha.Forward(graph, input);

            Assert.Equal(batch, output.Shape.D0);
            Assert.Equal(seq, output.Shape.D1);
            Assert.Equal(dModel, output.Shape.D2);
        }

        [Fact]
        public void MHA_NoNaNOrInf_WithCausalMask()
        {
            using var mha = new MultiHeadAttentionLayer(16, 4, causalMask: true);
            using var graph = new ComputationGraph();
            using var input = MakeInput(2, 6, 16, requiresGrad: false);
            using var output = mha.Forward(graph, input);

            foreach (var v in output.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v), "NaN in MHA output");
                Assert.False(float.IsInfinity(v), "Inf in MHA output");
            }
        }

        [Fact]
        public void MHA_NoNaNOrInf_WithoutCausalMask()
        {
            using var mha = new MultiHeadAttentionLayer(16, 4, causalMask: false);
            using var graph = new ComputationGraph();
            using var input = MakeInput(1, 5, 16, requiresGrad: false);
            using var output = mha.Forward(graph, input);

            foreach (var v in output.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v), "NaN in MHA output (no mask)");
                Assert.False(float.IsInfinity(v), "Inf in MHA output (no mask)");
            }
        }

        [Fact]
        public void MHA_CausalMask_FutureTokensDoNotAffectPast()
        {
            const int dModel = 8, nHeads = 2, seq = 4;
            using var mha = new MultiHeadAttentionLayer(dModel, nHeads, causalMask: true);

            var data1 = MakeRandom(1 * seq * dModel, seed: 42);
            using var input1 = MakeInputFromData(data1, 1, seq, dModel, requiresGrad: false);
            using var graph1 = new ComputationGraph();
            using var output1 = mha.Forward(graph1, input1);
            var out1 = output1.DataView.AsReadOnlySpan().ToArray();

            var data2 = (float[])data1.Clone();
            for (var d = 0; d < dModel; d++)
            {
                data2[3 * dModel + d] += 1f;
            }

            using var input2 = MakeInputFromData(data2, 1, seq, dModel, requiresGrad: false);
            using var graph2 = new ComputationGraph();
            using var output2 = mha.Forward(graph2, input2);
            var out2 = output2.DataView.AsReadOnlySpan().ToArray();

            for (var t = 0; t < 3; t++)
            {
                for (var d = 0; d < dModel; d++)
                {
                    var diff = MathF.Abs(out1[t * dModel + d] - out2[t * dModel + d]);
                    Assert.True(diff < Tolerance, $"Position {t} changed when future position 3 was perturbed. diff={diff:F6}");
                }
            }
        }

        [Fact]
        public void MHA_ParameterCount_IsCorrect()
        {
            // Per head:
            //   Wq/Wk/Wv: [dModel, dHead]
            //   Bq/Bk/Bv: [dHead]
            //   Wo:       [dHead, dModel]
            // Plus global Bo: [dModel]
            // Across all heads:
            //   4 * dModel * dModel + 4 * dModel
            const int dModel = 16, nHeads = 4;
            using var mha = new MultiHeadAttentionLayer(dModel, nHeads);

            var paramCount = mha.TrainableParameters().Sum(p => p.Shape.Size);
            var expected = 4 * dModel * dModel + 4 * dModel;

            Assert.Equal(expected, paramCount);
        }

        [Fact]
        public void MHA_DModelNotDivisibleByNHeads_Throws()
        {
            Assert.Throws<ArgumentException>(() => new MultiHeadAttentionLayer(dModel: 10, nHeads: 3));
        }

        [Fact]
        public void MHA_SaveLoad_ProducesSameOutput()
        {
            const int dModel = 8, nHeads = 2, batch = 1, seq = 3;

            using var mha1 = new MultiHeadAttentionLayer(dModel, nHeads);
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            mha1.Save(bw);

            ms.Position = 0;

            using var br = new BinaryReader(ms);
            using var mha2 = new MultiHeadAttentionLayer(dModel, nHeads);
            mha2.Load(br);

            var inputData = MakeRandom(batch * seq * dModel, seed: 99);

            using var graph1 = new ComputationGraph();
            using var input1 = MakeInputFromData(inputData, batch, seq, dModel, requiresGrad: false);
            using var output1 = mha1.Forward(graph1, input1);

            using var graph2 = new ComputationGraph();
            using var input2 = MakeInputFromData(inputData, batch, seq, dModel, requiresGrad: false);
            using var output2 = mha2.Forward(graph2, input2);

            var out1 = output1.DataView.AsReadOnlySpan();
            var out2 = output2.DataView.AsReadOnlySpan();

            for (var i = 0; i < out1.Length; i++)
            {
                Assert.True(MathF.Abs(out1[i] - out2[i]) < Tolerance, $"Mismatch at [{i}] after load: {out1[i]:F6} vs {out2[i]:F6}");
            }
        }

        [Fact]
        public void MHA_GPT1Scale_ForwardRunsWithoutException()
        {
            const int dModel = 768, nHeads = 12, batch = 1, seq = 8;

            using var mha = new MultiHeadAttentionLayer(dModel, nHeads, causalMask: true);
            using var graph = new ComputationGraph();
            using var input = MakeInput(batch, seq, dModel, requiresGrad: false);
            using var output = mha.Forward(graph, input);

            Assert.Equal(batch, output.Shape.D0);
            Assert.Equal(seq, output.Shape.D1);
            Assert.Equal(dModel, output.Shape.D2);

            foreach (var v in output.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v), "NaN in GPT-1 scale MHA output");
                Assert.False(float.IsInfinity(v), "Inf in GPT-1 scale MHA output");
            }
        }

        private static AutogradNode MakeInput(int b, int t, int d, bool requiresGrad)
        {
            var data = MakeRandom(b * t * d, seed: 42);
            return MakeInputFromData(data, b, t, d, requiresGrad);
        }

        private static AutogradNode MakeInputFromData(float[] data, int b, int t, int d, bool requiresGrad)
        {
            var storage = new TensorStorage<float>(data.Length, clearMemory: false);
            data.AsSpan().CopyTo(storage.AsSpan());
            return new AutogradNode(storage, new TensorShape(b, t, d), requiresGrad);
        }

        private static float[] MakeRandom(int size, int seed)
        {
            var rng = new Random(seed);
            var data = new float[size];

            for (var i = 0; i < size; i++)
            {
                data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
            }

            return data;
        }
    }
}
