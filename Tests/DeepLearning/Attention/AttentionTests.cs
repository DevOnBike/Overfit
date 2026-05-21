// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.DeepLearning.Attention
{
    /// <summary>
    /// Tests for Scaled Dot-Product Attention — core Transformer primitive.
    ///
    /// Tests cover:
    ///   - Output shape correctness
    ///   - Causal mask (future positions get zero attention)
    ///   - Attention weights sum to 1 per row (softmax property)
    ///   - Self-attention: identical Q/K/V attends to itself
    ///   - Numerical gradient check for Q, K, V
    ///   - ScaledDotProductAttentionLayer forward shape
    /// </summary>
    public class AttentionTests
    {
        private const float Tolerance = 1e-3f;

        // ─────────────────────────────────────────────────────────────────────
        // Shape tests
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void SDPA_OutputShape_IsCorrect()
        {
            // [B=2, T=4, dk=8] Q,K ; [B=2, T=4, dv=16] V → output [B=2, T=4, dv=16]
            using var g = new ComputationGraph();
            var (q, k, v) = MakeQKV(batch: 2, seq: 4, dk: 8, dv: 16);
            using var q_ = q; using var k_ = k; using var v_ = v;

            using var out_ = TensorMath.ScaledDotProductAttention(g, q, k, v, causalMask: false);

            Assert.Equal(2, out_.Shape.D0);
            Assert.Equal(4, out_.Shape.D1);
            Assert.Equal(16, out_.Shape.D2);
        }

        [Fact]
        public void SDPA_NoNaNOrInf_InOutput()
        {
            using var g = new ComputationGraph();
            var (q, k, v) = MakeQKV(batch: 1, seq: 6, dk: 4, dv: 4);
            using var q_ = q; using var k_ = k; using var v_ = v;

            using var out_ = TensorMath.ScaledDotProductAttention(g, q, k, v, causalMask: true);

            foreach (var val in out_.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(val), "NaN in SDPA output");
                Assert.False(float.IsInfinity(val), "Inf in SDPA output");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Causal mask
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void SDPA_CausalMask_FuturePositionsGetZeroAttention()
        {
            // With identical Q and K, without mask position 0 attends uniformly.
            // With causal mask, position 0 can only attend to position 0 (self).
            // So output[0] = V[0].
            const int seq = 4, dk = 4, dv = 4;

            using var g = new ComputationGraph();

            // V: each row is a distinct one-hot-like pattern
            var vData = new float[seq * dv];
            for (var t = 0; t < seq; t++)
            {
                vData[t * dv + t] = 1f; // row t has 1 at position t
            }

            // Q = K = ones (uniform attention without mask)
            var qData = new float[seq * dk]; qData.AsSpan().Fill(1f);
            var kData = new float[seq * dk]; kData.AsSpan().Fill(1f);

            using var q = MakeNode(qData, 1, seq, dk, true);
            using var k = MakeNode(kData, 1, seq, dk, true);
            using var v = MakeNode(vData, 1, seq, dv, true);

            using var out_ = TensorMath.ScaledDotProductAttention(g, q, k, v, causalMask: true);
            var outS = out_.DataView.AsReadOnlySpan();

            // Position 0: can only attend to position 0 → output[0] = V[0] = [1,0,0,0]
            Assert.True(MathF.Abs(outS[0] - 1f) < Tolerance, $"out[0,0,0] = {outS[0]}, expected 1");
            Assert.True(MathF.Abs(outS[1] - 0f) < Tolerance, $"out[0,0,1] = {outS[1]}, expected 0");
            Assert.True(MathF.Abs(outS[2] - 0f) < Tolerance, $"out[0,0,2] = {outS[2]}, expected 0");
            Assert.True(MathF.Abs(outS[3] - 0f) < Tolerance, $"out[0,0,3] = {outS[3]}, expected 0");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Softmax property
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void SDPA_NoCausalMask_AttentionWeightsSumToOne()
        {
            // Without causal mask: each row of attention matrix sums to 1 (softmax).
            // We verify via output: if V is all-ones, output should be all-ones.
            const int seq = 5, dk = 4, dv = 3;

            using var g = new ComputationGraph();

            var qData = new float[seq * dk]; new Random(42).NextFloats(qData);
            var kData = new float[seq * dk]; new Random(13).NextFloats(kData);
            var vData = new float[seq * dv]; vData.AsSpan().Fill(1f); // all-ones V

            using var q = MakeNode(qData, 1, seq, dk, false);
            using var k = MakeNode(kData, 1, seq, dk, false);
            using var v = MakeNode(vData, 1, seq, dv, false);

            using var out_ = TensorMath.ScaledDotProductAttention(g, q, k, v, causalMask: false);
            var outS = out_.DataView.AsReadOnlySpan();

            // Each output element should be 1.0 (weighted sum of all-ones V)
            foreach (var val in outS)
            {
                Assert.True(MathF.Abs(val - 1f) < Tolerance,
                    $"Expected 1.0 (attention sums to 1), got {val:F6}");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Numerical gradient check
        // ─────────────────────────────────────────────────────────────────────

        [Theory]
        [InlineData("Q")]
        [InlineData("K")]
        [InlineData("V")]
        public void SDPA_NumericalGradient_MatchesAnalytical(string which)
        {
            const int batch = 1, seq = 3, dk = 4, dv = 4;
            var rng = new Random(42);
            var eps = 1e-3f;
            var idx = 2; // element to perturb

            var qData = new float[batch * seq * dk];
            var kData = new float[batch * seq * dk];
            var vData = new float[batch * seq * dv];
            rng.NextFloats(qData); rng.NextFloats(kData); rng.NextFloats(vData);

            Func<float[], float[], float[], float> lossFunc = (qD, kD, vD) =>
            {
                using var g = new ComputationGraph();
                using var q2 = MakeNode(qD, batch, seq, dk, true);
                using var k2 = MakeNode(kD, batch, seq, dk, true);
                using var v2 = MakeNode(vD, batch, seq, dv, true);
                using var o2 = TensorMath.ScaledDotProductAttention(g, q2, k2, v2, causalMask: true);
                var sum = 0f;
                foreach (var x in o2.DataView.AsReadOnlySpan())
                {
                    sum += x;
                }
                return sum;
            };

            // Numerical
            var perturbedPlus = which == "Q" ? (float[])qData.Clone() :
                                     which == "K" ? (float[])kData.Clone() : (float[])vData.Clone();
            var perturbedMinus = which == "Q" ? (float[])qData.Clone() :
                                     which == "K" ? (float[])kData.Clone() : (float[])vData.Clone();
            perturbedPlus[idx] += eps;
            perturbedMinus[idx] -= eps;

            var lossPlus = which == "Q" ? lossFunc(perturbedPlus, kData, vData) :
                            which == "K" ? lossFunc(qData, perturbedPlus, vData) :
                                           lossFunc(qData, kData, perturbedPlus);
            var lossMinus = which == "Q" ? lossFunc(perturbedMinus, kData, vData) :
                            which == "K" ? lossFunc(qData, perturbedMinus, vData) :
                                           lossFunc(qData, kData, perturbedMinus);

            var numerical = (lossPlus - lossMinus) / (2 * eps);

            // Analytical
            using var gA = new ComputationGraph();
            using var qA = MakeNode((float[])qData.Clone(), batch, seq, dk, true);
            using var kA = MakeNode((float[])kData.Clone(), batch, seq, dk, true);
            using var vA = MakeNode((float[])vData.Clone(), batch, seq, dv, true);
            using var oA = TensorMath.ScaledDotProductAttention(gA, qA, kA, vA, causalMask: true);

            oA.GradView.AsSpan().Fill(1f); // seed = sum loss
            gA.Backward(oA);

            var analytical = which == "Q" ? qA.GradView.AsReadOnlySpan()[idx] :
                             which == "K" ? kA.GradView.AsReadOnlySpan()[idx] :
                                            vA.GradView.AsReadOnlySpan()[idx];

            var relErr = MathF.Abs(analytical - numerical) / (MathF.Abs(numerical) + 1e-7f);

            Assert.True(relErr < 0.03f,
                $"d{which}[{idx}]: analytical={analytical:F6}, numerical={numerical:F6}, relErr={relErr:F4}");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Flattened ([B*T, d]) overload parity
        // ─────────────────────────────────────────────────────────────────────

        [Theory]
        [InlineData(3, 5, 8, 8)]      // work = 600 — sub-threshold, sequential path
        [InlineData(4, 64, 16, 16)]   // work ≈ 1.0M — supra-threshold, parallel path
        public void SDPA_FlattenedOverload_MatchesThreeDimensional(
            int batch, int seq, int dk, int dv)
        {
            // The 2-D [B*T,d] overload and the 3-D [B,T,d] overload share a
            // byte-identical row-major layout and the same flat kernels, so
            // forward output AND backward gradients must be BIT-identical.
            var rng = new Random(7);
            var qData = new float[batch * seq * dk]; rng.NextFloats(qData);
            var kData = new float[batch * seq * dk]; rng.NextFloats(kData);
            var vData = new float[batch * seq * dv]; rng.NextFloats(vData);

            // 3-D reference path.
            using var g3 = new ComputationGraph();
            using var q3 = MakeNode((float[])qData.Clone(), batch, seq, dk, true);
            using var k3 = MakeNode((float[])kData.Clone(), batch, seq, dk, true);
            using var v3 = MakeNode((float[])vData.Clone(), batch, seq, dv, true);
            using var o3 = TensorMath.ScaledDotProductAttention(g3, q3, k3, v3, causalMask: true);
            o3.GradView.AsSpan().Fill(1f);
            g3.Backward(o3);

            // 2-D flattened path — same data, [B*T, d] node shapes.
            using var g2 = new ComputationGraph();
            using var q2 = MakeNode2D((float[])qData.Clone(), batch * seq, dk, true);
            using var k2 = MakeNode2D((float[])kData.Clone(), batch * seq, dk, true);
            using var v2 = MakeNode2D((float[])vData.Clone(), batch * seq, dv, true);
            using var o2 = TensorMath.ScaledDotProductAttention(g2, q2, k2, v2, batch, seq, causalMask: true);
            o2.GradView.AsSpan().Fill(1f);
            g2.Backward(o2);

            Assert.Equal(batch * seq, o2.Shape.D0);
            Assert.Equal(dv, o2.Shape.D1);

            AssertBitIdentical(o3.DataView.AsReadOnlySpan(), o2.DataView.AsReadOnlySpan(), "output");
            AssertBitIdentical(q3.GradView.AsReadOnlySpan(), q2.GradView.AsReadOnlySpan(), "dQ");
            AssertBitIdentical(k3.GradView.AsReadOnlySpan(), k2.GradView.AsReadOnlySpan(), "dK");
            AssertBitIdentical(v3.GradView.AsReadOnlySpan(), v2.GradView.AsReadOnlySpan(), "dV");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Layer-level test
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void SDPALayer_Forward_OutputShapeIsCorrect()
        {
            const int dModel = 16, dk = 8, dv = 8, batch = 2, seq = 5;

            using var layer = new ScaledDotProductAttentionLayer(dModel, dk, dv, causalMask: true);
            using var graph = new ComputationGraph();

            var storage = new TensorStorage<float>(batch * seq * dModel, clearMemory: false);
            storage.AsSpan().Fill(0.1f);
            using var input = new AutogradNode(storage, new TensorShape(batch, seq, dModel), requiresGrad: true);
            using var output = layer.Forward(graph, input);

            Assert.Equal(batch, output.Shape.D0);
            Assert.Equal(seq, output.Shape.D1);
            Assert.Equal(dModel, output.Shape.D2);
        }

        [Fact]
        public void SDPALayer_Forward_NoNaNOrInf()
        {
            const int dModel = 8, dk = 4, dv = 4, batch = 1, seq = 6;

            using var layer = new ScaledDotProductAttentionLayer(dModel, dk, dv, causalMask: true);
            using var graph = new ComputationGraph();

            var storage = new TensorStorage<float>(batch * seq * dModel, clearMemory: false);
            new Random(42).NextFloats(storage.AsSpan().ToArray()).AsSpan().CopyTo(storage.AsSpan());
            using var input = new AutogradNode(storage, new TensorShape(batch, seq, dModel), requiresGrad: true);
            using var output = layer.Forward(graph, input);

            foreach (var v in output.DataView.AsReadOnlySpan())
            {
                Assert.False(float.IsNaN(v), "NaN in SDPA layer output");
                Assert.False(float.IsInfinity(v), "Inf in SDPA layer output");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helpers
        // ─────────────────────────────────────────────────────────────────────

        private static (AutogradNode q, AutogradNode k, AutogradNode v) MakeQKV(
            int batch, int seq, int dk, int dv)
        {
            var rng = new Random(42);
            var qData = new float[batch * seq * dk]; rng.NextFloats(qData);
            var kData = new float[batch * seq * dk]; rng.NextFloats(kData);
            var vData = new float[batch * seq * dv]; rng.NextFloats(vData);

            return (MakeNode(qData, batch, seq, dk, true),
                    MakeNode(kData, batch, seq, dk, true),
                    MakeNode(vData, batch, seq, dv, true));
        }

        private static AutogradNode MakeNode(float[] data, int b, int t, int d, bool requiresGrad)
        {
            var storage = new TensorStorage<float>(data.Length, clearMemory: false);
            data.AsSpan().CopyTo(storage.AsSpan());
            return new AutogradNode(storage, new TensorShape(b, t, d), requiresGrad);
        }

        private static AutogradNode MakeNode2D(float[] data, int rows, int d, bool requiresGrad)
        {
            var storage = new TensorStorage<float>(data.Length, clearMemory: false);
            data.AsSpan().CopyTo(storage.AsSpan());
            return new AutogradNode(storage, new TensorShape(rows, d), requiresGrad);
        }

        private static void AssertBitIdentical(
            ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string label)
        {
            Assert.Equal(expected.Length, actual.Length);

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.True(
                    expected[i] == actual[i],
                    $"{label}[{i}]: 3-D={expected[i]} != 2-D={actual[i]}");
            }
        }
    }
}
