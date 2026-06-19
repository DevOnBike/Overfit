// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;
using Ops = DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.Core.Autograd
{
    /// <summary>
    /// Autograd RoPE op (<see cref="Ops.TensorMath.Rope"/>): the forward is bit-faithful to the
    /// inference kernel <see cref="RopeKernel.Apply(System.Span{float}, RopeTable, int)"/>
    /// (adjacent-pair / GGUF layout — the layout-fidelity guarantee that lets a GGUF base rotate
    /// identically in training and inference), and the backward (inverse rotation) passes a central
    /// finite-difference check end-to-end through MSE.
    /// </summary>
    public sealed class RopeTests
    {
        private const int T = 5;        // tokens / positions
        private const int NHeads = 3;
        private const int HeadDim = 8;  // even
        private const int HalfDim = HeadDim / 2;
        private const int D = NHeads * HeadDim;
        private const float Theta = 10_000f;

        private readonly ITestOutputHelper _out;
        public RopeTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Forward_MatchesInferenceKernel_AdjacentPairLayout()
        {
            var table = new RopeTable(maxSequenceLength: T, headDimension: HeadDim, theta: Theta);
            var x = RandomVector(T * D, seed: 1);
            var (cos, sin) = BuildCosSin(table);

            using var xData = Storage(x);
            using var cData = Storage(cos);
            using var sData = Storage(sin);
            using var xNode = new AutogradNode(xData, new TensorShape(T, D), requiresGrad: true);
            using var cNode = new AutogradNode(cData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var sNode = new AutogradNode(sData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            using var y = Ops.TensorMath.Rope(graph, xNode, cNode, sNode);
            var ys = y.DataView.AsReadOnlySpan();

            // Reference: rotate each head independently with the SAME inference kernel.
            var reference = (float[])x.Clone();
            for (var t = 0; t < T; t++)
            {
                for (var h = 0; h < NHeads; h++)
                {
                    var slice = reference.AsSpan(t * D + h * HeadDim, HeadDim);
                    RopeKernel.Apply(slice, table, t);
                }
            }

            double maxAbs = 0;
            for (var i = 0; i < T * D; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(reference[i] - ys[i]));
            }
            _out.WriteLine($"RoPE forward maxAbs vs RopeKernel: {maxAbs:E3}");
            Assert.True(maxAbs < 1e-5, $"RoPE forward differs from inference kernel: {maxAbs:E3}");
        }

        [Fact]
        public void Forward_MatchesInferenceKernel_SplitHalfLayout()
        {
            // Qwen2/2.5/3 GGUFs use split-half (HF rotate_half) — RopeSplitHalf=true.
            var table = new RopeTable(maxSequenceLength: T, headDimension: HeadDim, theta: Theta, splitHalf: true);
            var x = RandomVector(T * D, seed: 11);
            var (cos, sin) = BuildCosSin(table);

            using var xData = Storage(x);
            using var cData = Storage(cos);
            using var sData = Storage(sin);
            using var xNode = new AutogradNode(xData, new TensorShape(T, D), requiresGrad: true);
            using var cNode = new AutogradNode(cData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var sNode = new AutogradNode(sData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            using var y = Ops.TensorMath.Rope(graph, xNode, cNode, sNode, splitHalf: true);
            var ys = y.DataView.AsReadOnlySpan();

            var reference = (float[])x.Clone();
            for (var t = 0; t < T; t++)
            {
                for (var h = 0; h < NHeads; h++)
                {
                    RopeKernel.Apply(reference.AsSpan(t * D + h * HeadDim, HeadDim), table, t); // table.SplitHalf=true
                }
            }

            double maxAbs = 0;
            for (var i = 0; i < T * D; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(reference[i] - ys[i]));
            }
            _out.WriteLine($"RoPE split-half forward maxAbs vs RopeKernel: {maxAbs:E3}");
            Assert.True(maxAbs < 1e-5, $"split-half forward differs from inference kernel: {maxAbs:E3}");
        }

        [Fact]
        public void Backward_SplitHalf_PassesFiniteDifference()
        {
            var table = new RopeTable(maxSequenceLength: T, headDimension: HeadDim, theta: Theta, splitHalf: true);
            var x = RandomVector(T * D, seed: 12);
            var target = RandomVector(T * D, seed: 13);
            var (cos, sin) = BuildCosSin(table);

            using var xData = Storage(x);
            using var cData = Storage(cos);
            using var sData = Storage(sin);
            using var tData = Storage(target);
            using var xNode = new AutogradNode(xData, new TensorShape(T, D), requiresGrad: true);
            using var cNode = new AutogradNode(cData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var sNode = new AutogradNode(sData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var tNode = new AutogradNode(tData, new TensorShape(T, D), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            graph.Reset();
            xNode.GradView.AsSpan().Clear();
            var y = Ops.TensorMath.Rope(graph, xNode, cNode, sNode, splitHalf: true);
            var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            graph.Backward(loss);
            var dxA = xNode.GradView.AsReadOnlySpan().ToArray();

            float LossAt()
            {
                graph.Reset();
                var yy = Ops.TensorMath.Rope(graph, xNode, cNode, sNode, splitHalf: true);
                var ll = Ops.TensorMath.MSELoss(graph, yy, tNode);
                return ll.DataView.AsReadOnlySpan()[0];
            }

            const float eps = 1e-3f;
            var xs = xData.AsSpan();
            double maxRel = 0;
            foreach (var idx in new[] { 0, 3, 7, 8, 15, D + 1, 2 * D + 5, T * D - 1 })
            {
                var orig = xs[idx];
                xs[idx] = orig + eps;
                var lp = LossAt();
                xs[idx] = orig - eps;
                var lm = LossAt();
                xs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var absDiff = Math.Abs(fd - dxA[idx]);
                var rel = absDiff / Math.Max(1e-3, Math.Abs(dxA[idx]));
                if (absDiff > 5e-4)
                {
                    maxRel = Math.Max(maxRel, rel);
                } // skip entries below the FD noise floor
            }
            Assert.True(maxRel < 2e-2, $"split-half RoPE finite-difference mismatch, maxRel {maxRel:E3}");
        }

        [Fact]
        public void Backward_PassesFiniteDifference()
        {
            var table = new RopeTable(maxSequenceLength: T, headDimension: HeadDim, theta: Theta);
            var x = RandomVector(T * D, seed: 2);
            var target = RandomVector(T * D, seed: 3);
            var (cos, sin) = BuildCosSin(table);

            using var xData = Storage(x);
            using var cData = Storage(cos);
            using var sData = Storage(sin);
            using var tData = Storage(target);
            using var xNode = new AutogradNode(xData, new TensorShape(T, D), requiresGrad: true);
            using var cNode = new AutogradNode(cData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var sNode = new AutogradNode(sData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var tNode = new AutogradNode(tData, new TensorShape(T, D), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            graph.Reset();
            xNode.GradView.AsSpan().Clear();
            var y = Ops.TensorMath.Rope(graph, xNode, cNode, sNode);
            var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            graph.Backward(loss);
            var dxA = xNode.GradView.AsReadOnlySpan().ToArray();

            float LossAt()
            {
                graph.Reset();
                var yy = Ops.TensorMath.Rope(graph, xNode, cNode, sNode);
                var ll = Ops.TensorMath.MSELoss(graph, yy, tNode);
                return ll.DataView.AsReadOnlySpan()[0];
            }

            const float eps = 1e-3f;
            var xs = xData.AsSpan();
            double maxRel = 0;
            foreach (var idx in new[] { 0, 3, 7, 8, 15, D + 1, 2 * D + 5, T * D - 1 })
            {
                var orig = xs[idx];
                xs[idx] = orig + eps;
                var lp = LossAt();
                xs[idx] = orig - eps;
                var lm = LossAt();
                xs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var an = dxA[idx];
                var absDiff = Math.Abs(fd - an);
                var rel = absDiff / Math.Max(1e-3, Math.Abs(an));
                if (absDiff > 5e-4)
                {
                    maxRel = Math.Max(maxRel, rel);
                } // skip entries below the FD noise floor
                _out.WriteLine($"  dX[{idx}]: analytic {an:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 2e-2, $"RoPE finite-difference mismatch, maxRel {maxRel:E3}");
        }

        [Fact]
        public void ForwardThenInverse_RecoversInput()
        {
            // The backward op is the inverse rotation; seeding dOut = forward(x) and running the
            // backward kernel must recover x exactly (orthogonality of the rotation).
            var table = new RopeTable(maxSequenceLength: T, headDimension: HeadDim, theta: Theta);
            var x = RandomVector(T * D, seed: 4);
            var (cos, sin) = BuildCosSin(table);

            using var xData = Storage(x);
            using var cData = Storage(cos);
            using var sData = Storage(sin);
            using var xNode = new AutogradNode(xData, new TensorShape(T, D), requiresGrad: true);
            using var cNode = new AutogradNode(cData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var sNode = new AutogradNode(sData, new TensorShape(T, HalfDim), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 16);

            using var y = Ops.TensorMath.Rope(graph, xNode, cNode, sNode);

            // Put forward(x) into y's grad, zero x's grad, run backward = inverse rotation.
            y.GradView.AsSpan().Clear();
            y.DataView.AsReadOnlySpan().CopyTo(y.GradView.AsSpan());
            xNode.GradView.AsSpan().Clear();
            Ops.TensorMath.RopeBackward(xNode, y, cNode, sNode, splitHalf: false);

            var recovered = xNode.GradView.AsReadOnlySpan();
            double maxAbs = 0;
            for (var i = 0; i < T * D; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(x[i] - recovered[i]));
            }
            _out.WriteLine($"RoPE inverse round-trip maxAbs: {maxAbs:E3}");
            Assert.True(maxAbs < 1e-5, $"inverse rotation did not recover input: {maxAbs:E3}");
        }

        // ── helpers ──

        private static (float[] cos, float[] sin) BuildCosSin(RopeTable table)
        {
            var cos = new float[T * HalfDim];
            var sin = new float[T * HalfDim];
            for (var t = 0; t < T; t++)
            {
                table.CosAt(t).CopyTo(cos.AsSpan(t * HalfDim, HalfDim));
                table.SinAt(t).CopyTo(sin.AsSpan(t * HalfDim, HalfDim));
            }
            return (cos, sin);
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
