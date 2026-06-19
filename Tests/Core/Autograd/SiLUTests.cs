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
    /// Autograd SiLU / swish op (<see cref="Ops.TensorMath.SiLU"/>): forward matches the
    /// scalar reference <c>x·σ(x)</c> (small AND parallel paths), and the backward passes a
    /// central finite-difference check end-to-end through an MSE loss.
    /// </summary>
    public sealed class SiLUTests
    {
        private readonly ITestOutputHelper _out;
        public SiLUTests(ITestOutputHelper output) => _out = output;

        [Theory]
        [InlineData(256)]    // sequential path
        [InlineData(8192)]   // parallel path (> ParallelThreshold 4096)
        public void Forward_MatchesScalarReference(int n)
        {
            var x = RandomVector(n, seed: 1);
            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(n), requiresGrad: true);
            var graph = new ComputationGraph(1 << 18);

            using var y = Ops.TensorMath.SiLU(graph, xNode);

            var ys = y.DataView.AsReadOnlySpan();
            double maxAbs = 0;
            for (var i = 0; i < n; i++)
            {
                var s = 1f / (1f + MathF.Exp(-x[i]));
                maxAbs = Math.Max(maxAbs, Math.Abs(x[i] * s - ys[i]));
            }
            _out.WriteLine($"SiLU forward maxAbs (n={n}): {maxAbs:E3}");
            Assert.True(maxAbs < 1e-5, $"SiLU forward differs from reference: {maxAbs:E3}");
        }

        [Fact]
        public void Backward_PassesFiniteDifference()
        {
            const int n = 64;
            var x = RandomVector(n, seed: 2);
            var target = RandomVector(n, seed: 3);

            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(n), requiresGrad: true);
            using var tData = Storage(target);
            using var tNode = new AutogradNode(tData, new TensorShape(n), requiresGrad: false);
            var graph = new ComputationGraph(1 << 16);

            graph.Reset();
            xNode.GradView.AsSpan().Clear();
            var y = Ops.TensorMath.SiLU(graph, xNode);
            var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            graph.Backward(loss);
            var dxA = xNode.GradView.AsReadOnlySpan().ToArray();

            float LossAt()
            {
                graph.Reset();
                var yy = Ops.TensorMath.SiLU(graph, xNode);
                var ll = Ops.TensorMath.MSELoss(graph, yy, tNode);
                return ll.DataView.AsReadOnlySpan()[0];
            }

            const float eps = 1e-3f;
            var xs = xData.AsSpan();
            double maxRel = 0;
            foreach (var idx in new[] { 0, 7, 13, 31, 50, 63 })
            {
                var orig = xs[idx];
                xs[idx] = orig + eps;
                var lp = LossAt();
                xs[idx] = orig - eps;
                var lm = LossAt();
                xs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var an = dxA[idx];
                var rel = Math.Abs(fd - an) / Math.Max(1e-4, Math.Abs(an));
                maxRel = Math.Max(maxRel, rel);
                _out.WriteLine($"  x[{idx}]: analytic {an:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 2e-2, $"SiLU finite-difference mismatch, maxRel {maxRel:E3}");
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
                v[i] = (float)(r.NextDouble() * 4 - 2);
            }
            return v;
        }
    }
}
