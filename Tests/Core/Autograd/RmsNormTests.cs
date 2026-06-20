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
    /// Autograd RMSNorm op (<see cref="Ops.TensorMath.RmsNorm"/>): forward matches the
    /// per-row reference <c>x / sqrt(mean(x²)+eps) · γ</c> (sequential AND parallel paths),
    /// and both gradients (dInput, dGamma) pass a central finite-difference check through MSE.
    /// </summary>
    public sealed class RmsNormTests
    {
        private const float Eps = 1e-6f;
        private readonly ITestOutputHelper _out;
        public RmsNormTests(ITestOutputHelper output) => _out = output;

        [Theory]
        [InlineData(4, 32)]     // 128 elems  → sequential
        [InlineData(128, 64)]   // 8192 elems → parallel (> 4096)
        public void Forward_MatchesReference(int rows, int c)
        {
            var x = RandomVector(rows * c, seed: 1);
            var gamma = RandomVector(c, seed: 2);

            using var xData = Storage(x);
            using var gData = Storage(gamma);
            using var xNode = new AutogradNode(xData, new TensorShape(rows, c), requiresGrad: true);
            using var gNode = new AutogradNode(gData, new TensorShape(c), requiresGrad: true);
            using var graph = new ComputationGraph(1 << 18);

            using var y = Ops.TensorMath.RmsNorm(graph, xNode, gNode, Eps);

            var ys = y.DataView.AsReadOnlySpan();
            double maxAbs = 0;
            for (var r = 0; r < rows; r++)
            {
                var ms = 0f;
                for (var i = 0; i < c; i++)
                {
                    ms += x[r * c + i] * x[r * c + i];
                }
                ms /= c;
                var inv = 1f / MathF.Sqrt(ms + Eps);
                for (var i = 0; i < c; i++)
                {
                    var expected = x[r * c + i] * inv * gamma[i];
                    maxAbs = Math.Max(maxAbs, Math.Abs(expected - ys[r * c + i]));
                }
            }
            _out.WriteLine($"RMSNorm forward maxAbs (rows={rows}, c={c}): {maxAbs:E3}");
            Assert.True(maxAbs < 1e-4, $"RMSNorm forward differs from reference: {maxAbs:E3}");
        }

        [Fact]
        public void InputGradient_PassesFiniteDifference()
        {
            const int rows = 6, c = 16;
            var x = RandomVector(rows * c, seed: 3);
            var gamma = RandomVector(c, seed: 4);
            var target = RandomVector(rows * c, seed: 5);

            using var xData = Storage(x);
            using var gData = Storage(gamma);
            using var tData = Storage(target);
            using var xNode = new AutogradNode(xData, new TensorShape(rows, c), requiresGrad: true);
            using var gNode = new AutogradNode(gData, new TensorShape(c), requiresGrad: true);
            using var tNode = new AutogradNode(tData, new TensorShape(rows, c), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            var dxA = AnalyticGrad(graph, xNode, gNode, tNode, useInput: true);

            const float eps = 1e-3f;
            var xs = xData.AsSpan();
            double maxRel = 0;
            foreach (var idx in new[] { 0, 5, 16, 31, 50, 80, rows * c - 1 })
            {
                var orig = xs[idx];
                xs[idx] = orig + eps;
                var lp = LossAt(graph, xNode, gNode, tNode);
                xs[idx] = orig - eps;
                var lm = LossAt(graph, xNode, gNode, tNode);
                xs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var an = dxA[idx];
                // Mixed abs/rel tolerance: a central-difference estimate (eps=1e-3) can't resolve a gradient
                // near the FD noise floor, so skip the relative check for coordinates whose absolute mismatch
                // is below it — their relative error is meaningless and CPU-rounding dependent. A real gradient
                // bug shows a large absolute mismatch on the meaningful (~1e-2) entries, which this still catches.
                var absDiff = Math.Abs(fd - an);
                var rel = absDiff / Math.Max(1e-3, Math.Abs(an));
                if (absDiff > 5e-4)
                {
                    maxRel = Math.Max(maxRel, rel);
                }
                _out.WriteLine($"  dX[{idx}]: analytic {an:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 3e-2, $"RMSNorm dInput finite-difference mismatch, maxRel {maxRel:E3}");
        }

        [Fact]
        public void GammaGradient_PassesFiniteDifference()
        {
            const int rows = 6, c = 16;
            var x = RandomVector(rows * c, seed: 6);
            var gamma = RandomVector(c, seed: 7);
            var target = RandomVector(rows * c, seed: 8);

            using var xData = Storage(x);
            using var gData = Storage(gamma);
            using var tData = Storage(target);
            using var xNode = new AutogradNode(xData, new TensorShape(rows, c), requiresGrad: true);
            using var gNode = new AutogradNode(gData, new TensorShape(c), requiresGrad: true);
            using var tNode = new AutogradNode(tData, new TensorShape(rows, c), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            var dgA = AnalyticGrad(graph, xNode, gNode, tNode, useInput: false);

            const float eps = 1e-3f;
            var gs = gData.AsSpan();
            double maxRel = 0;
            for (var idx = 0; idx < c; idx++)
            {
                var orig = gs[idx];
                gs[idx] = orig + eps;
                var lp = LossAt(graph, xNode, gNode, tNode);
                gs[idx] = orig - eps;
                var lm = LossAt(graph, xNode, gNode, tNode);
                gs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var an = dgA[idx];
                // Mixed abs/rel tolerance: floor the denominator so coordinates whose analytic
                // gradient is ~0 don't blow the relative error up out of float FD precision.
                var rel = Math.Abs(fd - an) / Math.Max(1e-3, Math.Abs(an));
                maxRel = Math.Max(maxRel, rel);
                _out.WriteLine($"  dGamma[{idx}]: analytic {an:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 3e-2, $"RMSNorm dGamma finite-difference mismatch, maxRel {maxRel:E3}");
        }

        // ── helpers ──

        private static float[] AnalyticGrad(
            ComputationGraph graph, AutogradNode xNode, AutogradNode gNode, AutogradNode tNode, bool useInput)
        {
            graph.Reset();
            xNode.GradView.AsSpan().Clear();
            gNode.GradView.AsSpan().Clear();
            var y = Ops.TensorMath.RmsNorm(graph, xNode, gNode, Eps);
            var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            graph.Backward(loss);
            return (useInput ? xNode : gNode).GradView.AsReadOnlySpan().ToArray();
        }

        private static float LossAt(ComputationGraph graph, AutogradNode xNode, AutogradNode gNode, AutogradNode tNode)
        {
            graph.Reset();
            var yy = Ops.TensorMath.RmsNorm(graph, xNode, gNode, Eps);
            var ll = Ops.TensorMath.MSELoss(graph, yy, tNode);
            return ll.DataView.AsReadOnlySpan()[0];
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
