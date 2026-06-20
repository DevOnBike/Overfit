// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;
using Ops = DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.Core.Autograd
{
    /// <summary>
    /// Grouped-Query Attention (GQA). The KV-head broadcast op
    /// (<see cref="Ops.TensorMath.ExpandKvHeads"/>) lets the validated MHA SDPA kernel serve GQA:
    /// forward copies each KV head to its query-head group, and the backward (the GQA-specific
    /// gradient) sums each group's gradient into the shared KV head. Validated standalone (forward +
    /// finite-difference) and composed end-to-end with the 3-D SDPA on a real GQA shape
    /// (4 query heads : 2 KV heads, group size 2), FD-checking dQ, dK, dV.
    /// </summary>
    public sealed class GqaAttentionTests
    {
        private readonly ITestOutputHelper _out;
        public GqaAttentionTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Expand_Forward_BroadcastsKvHeadsToGroups()
        {
            const int kvHeads = 2, groupSize = 3, block = 5; // qHeads = 6
            var x = RandomVector(kvHeads * block, seed: 1);

            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(kvHeads, block), requiresGrad: true);
            using var graph = new ComputationGraph(1 << 16);

            using var y = Ops.TensorMath.ExpandKvHeads(graph, xNode, kvHeads, groupSize);

            Assert.Equal(kvHeads * groupSize, y.Shape.D0);
            var ys = y.DataView.AsReadOnlySpan();
            for (var qh = 0; qh < kvHeads * groupSize; qh++)
            {
                var kvh = qh / groupSize;
                for (var i = 0; i < block; i++)
                {
                    Assert.Equal(x[kvh * block + i], ys[qh * block + i]);
                }
            }
        }

        [Fact]
        public void Expand_Backward_SumsGroupGradientIntoSharedHead()
        {
            const int kvHeads = 2, groupSize = 2, block = 4; // qHeads = 4
            var x = RandomVector(kvHeads * block, seed: 2);
            var target = RandomVector(kvHeads * groupSize * block, seed: 3);

            using var xData = Storage(x);
            using var tData = Storage(target);
            using var xNode = new AutogradNode(xData, new TensorShape(kvHeads, block), requiresGrad: true);
            using var tNode = new AutogradNode(tData, new TensorShape(kvHeads * groupSize, block), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            var dxA = ExpandAnalyticGrad(graph, xNode, tNode, kvHeads, groupSize);

            const float eps = 1e-3f;
            var xs = xData.AsSpan();
            double maxRel = 0;
            for (var idx = 0; idx < kvHeads * block; idx++)
            {
                var orig = xs[idx];
                xs[idx] = orig + eps;
                var lp = ExpandLossAt(graph, xNode, tNode, kvHeads, groupSize);
                xs[idx] = orig - eps;
                var lm = ExpandLossAt(graph, xNode, tNode, kvHeads, groupSize);
                xs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var absDiff = Math.Abs(fd - dxA[idx]);
                var rel = absDiff / Math.Max(1e-3, Math.Abs(dxA[idx]));
                if (absDiff > 5e-4)
                {
                    maxRel = Math.Max(maxRel, rel);
                } // skip entries below the FD noise floor
            }
            _out.WriteLine($"ExpandKvHeads dInput FD maxRel: {maxRel:E3}");
            Assert.True(maxRel < 2e-2, $"ExpandKvHeads finite-difference mismatch, maxRel {maxRel:E3}");
        }

        [Fact]
        public void GqaAttention_FullPath_PassesFiniteDifference()
        {
            // Real GQA shape: 4 query heads, 2 KV heads (group size 2), T=4, dk=dv=6.
            const int nQHeads = 4, nKVHeads = 2, groupSize = nQHeads / nKVHeads;
            const int T = 4, d = 6;

            var q = RandomVector(nQHeads * T * d, seed: 4);
            var k = RandomVector(nKVHeads * T * d, seed: 5);
            var vv = RandomVector(nKVHeads * T * d, seed: 6);
            var target = RandomVector(nQHeads * T * d, seed: 7);

            using var qData = Storage(q);
            using var kData = Storage(k);
            using var vData = Storage(vv);
            using var tData = Storage(target);
            using var qNode = new AutogradNode(qData, new TensorShape(nQHeads, T, d), requiresGrad: true);
            using var kNode = new AutogradNode(kData, new TensorShape(nKVHeads, T, d), requiresGrad: true);
            using var vNode = new AutogradNode(vData, new TensorShape(nKVHeads, T, d), requiresGrad: true);
            using var tNode = new AutogradNode(tData, new TensorShape(nQHeads, T, d), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 18);

            float[] AnalyticGrad(AutogradNode wrt)
            {
                graph.Reset();
                qNode.GradView.AsSpan().Clear();
                kNode.GradView.AsSpan().Clear();
                vNode.GradView.AsSpan().Clear();
                var loss = GqaForward(graph, qNode, kNode, vNode, tNode, nKVHeads, groupSize);
                graph.Backward(loss);
                return wrt.GradView.AsReadOnlySpan().ToArray();
            }

            float LossAt()
            {
                graph.Reset();
                var loss = GqaForward(graph, qNode, kNode, vNode, tNode, nKVHeads, groupSize);
                return loss.DataView.AsReadOnlySpan()[0];
            }

            CheckFd("dQ", AnalyticGrad(qNode), qData, LossAt, new[] { 0, 7, T * d + 3, 3 * T * d + 5 });
            CheckFd("dK", AnalyticGrad(kNode), kData, LossAt, new[] { 0, 5, T * d + 2, nKVHeads * T * d - 1 });
            CheckFd("dV", AnalyticGrad(vNode), vData, LossAt, new[] { 1, 6, T * d + 4, nKVHeads * T * d - 2 });
        }

        // ── helpers ──

        private static AutogradNode GqaForward(
            ComputationGraph graph, AutogradNode q, AutogradNode k, AutogradNode v, AutogradNode target,
            int nKVHeads, int groupSize)
        {
            var kExp = Ops.TensorMath.ExpandKvHeads(graph, k, nKVHeads, groupSize);   // [nQHeads,T,d]
            var vExp = Ops.TensorMath.ExpandKvHeads(graph, v, nKVHeads, groupSize);
            var o = Ops.TensorMath.ScaledDotProductAttention(graph, q, kExp, vExp, causalMask: true);
            return Ops.TensorMath.MSELoss(graph, o, target);
        }

        private void CheckFd(string name, float[] analytic, TensorStorage<float> data, Func<float> lossAt, int[] indices)
        {
            const float eps = 1e-3f;
            var s = data.AsSpan();
            double maxRel = 0;
            foreach (var idx in indices)
            {
                var orig = s[idx];
                s[idx] = orig + eps;
                var lp = lossAt();
                s[idx] = orig - eps;
                var lm = lossAt();
                s[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var absDiff = Math.Abs(fd - analytic[idx]);
                var rel = absDiff / Math.Max(1e-3, Math.Abs(analytic[idx]));
                if (absDiff > 5e-4)
                {
                    maxRel = Math.Max(maxRel, rel);
                } // skip entries below the FD noise floor
                _out.WriteLine($"  {name}[{idx}]: analytic {analytic[idx]:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 3e-2, $"{name} finite-difference mismatch, maxRel {maxRel:E3}");
        }

        private static float[] ExpandAnalyticGrad(
            ComputationGraph graph, AutogradNode xNode, AutogradNode tNode, int kvHeads, int groupSize)
        {
            graph.Reset();
            xNode.GradView.AsSpan().Clear();
            var y = Ops.TensorMath.ExpandKvHeads(graph, xNode, kvHeads, groupSize);
            var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            graph.Backward(loss);
            return xNode.GradView.AsReadOnlySpan().ToArray();
        }

        private static float ExpandLossAt(
            ComputationGraph graph, AutogradNode xNode, AutogradNode tNode, int kvHeads, int groupSize)
        {
            graph.Reset();
            var y = Ops.TensorMath.ExpandKvHeads(graph, xNode, kvHeads, groupSize);
            var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            return loss.DataView.AsReadOnlySpan()[0];
        }

        private static TensorStorage<float> Storage(float[] data)
        {
            var s = new TensorStorage<float>(data.Length, clearMemory: false);
            data.CopyTo(s.AsSpan());
            return s;
        }

        private static float[] RandomVector(int n, int seed)
        {
            var r = new Random(seed);
            var v = new float[n];
            for (var i = 0; i < n; i++)
            {
                v[i] = (float)(r.NextDouble() * 2 - 1);
            }
            return v;
        }
    }
}
