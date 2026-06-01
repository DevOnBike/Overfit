// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;
using Ops = DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.Core.Autograd
{
    /// <summary>
    /// Validates the QLoRA base op <see cref="ComputationGraph.FrozenQuantizedLinear"/>: forward
    /// matches a full F32-dequant matmul, the input gradient matches the analytic
    /// <c>dx = dyᵀ·dequant(W)</c>, and a finite-difference check confirms the backward end-to-end.
    /// The frozen quantized weight never receives a gradient (it is not an <c>AutogradNode</c>).
    /// </summary>
    public sealed class FrozenQuantizedLinearTests
    {
        private readonly ITestOutputHelper _out;
        public FrozenQuantizedLinearTests(ITestOutputHelper output) => _out = output;

        private const int N = 4;    // batch
        private const int K = 256;  // input features (1 Q4_K super-block / row)
        private const int M = 8;    // outputs

        [Fact]
        public void Forward_MatchesF32DequantMatmul()
        {
            var weight = BuildRandomQ4K(seed: 1);
            var wF32 = DequantizeAll(weight);
            var x = RandomVector(N * K, seed: 2);

            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(N, K), requiresGrad: true);
            var graph = new ComputationGraph();
            using var y = graph.FrozenQuantizedLinear(xNode, weight);

            var ys = y.DataView.AsReadOnlySpan();
            var maxAbs = 0.0;
            for (var b = 0; b < N; b++)
            {
                for (var o = 0; o < M; o++)
                {
                    var expected = 0f;
                    for (var i = 0; i < K; i++) { expected += wF32[o * K + i] * x[b * K + i]; }
                    maxAbs = Math.Max(maxAbs, Math.Abs(expected - ys[b * M + o]));
                }
            }
            _out.WriteLine($"forward maxAbs vs F32-dequant: {maxAbs:E3}");
            Assert.True(maxAbs < 1e-3, $"forward differs from F32-dequant matmul: {maxAbs:E3}");
        }

        [Fact]
        public void InputGradient_MatchesAnalytic_AndWeightIsFrozen()
        {
            var weight = BuildRandomQ4K(seed: 3);
            var wF32 = DequantizeAll(weight);
            var x = RandomVector(N * K, seed: 4);
            var target = RandomVector(N * M, seed: 5);

            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(N, K), requiresGrad: true);
            using var tData = Storage(target);
            using var tNode = new AutogradNode(tData, new TensorShape(N, M), requiresGrad: false);

            var graph = new ComputationGraph();
            xNode.GradView.AsSpan().Clear();
            using var y = graph.FrozenQuantizedLinear(xNode, weight);
            using var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            graph.Backward(loss);

            // dy = ∂loss/∂y, read straight from y's grad after backward (whatever MSE's normalization).
            var dy = y.GradView.AsReadOnlySpan().ToArray();
            var dx = xNode.GradView.AsReadOnlySpan();

            // analytic: dx[b,i] = Σ_o dy[b,o] · W[o,i]
            var maxAbs = 0.0;
            for (var b = 0; b < N; b++)
            {
                for (var i = 0; i < K; i++)
                {
                    var expected = 0f;
                    for (var o = 0; o < M; o++) { expected += dy[b * M + o] * wF32[o * K + i]; }
                    maxAbs = Math.Max(maxAbs, Math.Abs(expected - dx[b * K + i]));
                }
            }
            _out.WriteLine($"dInput maxAbs vs analytic dyᵀ·W: {maxAbs:E3}");
            Assert.True(maxAbs < 1e-3, $"dInput differs from analytic: {maxAbs:E3}");
        }

        [Fact]
        public void InputGradient_PassesFiniteDifference()
        {
            var weight = BuildRandomQ4K(seed: 6);
            var x = RandomVector(N * K, seed: 7);
            var target = RandomVector(N * M, seed: 8);

            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(N, K), requiresGrad: true);
            using var tData = Storage(target);
            using var tNode = new AutogradNode(tData, new TensorShape(N, M), requiresGrad: false);
            var graph = new ComputationGraph();

            // analytic gradient
            graph.Reset();
            xNode.GradView.AsSpan().Clear();
            var y = graph.FrozenQuantizedLinear(xNode, weight);
            var loss = Ops.TensorMath.MSELoss(graph, y, tNode);
            graph.Backward(loss);
            var dxA = xNode.GradView.AsReadOnlySpan().ToArray();

            float LossAt()
            {
                graph.Reset();
                var yy = graph.FrozenQuantizedLinear(xNode, weight);
                var ll = Ops.TensorMath.MSELoss(graph, yy, tNode);
                return ll.DataView.AsReadOnlySpan()[0];
            }

            const float eps = 1e-3f;
            var xs = xData.AsSpan();
            double maxRel = 0;
            // central-difference a spread of coordinates
            foreach (var idx in new[] { 0, 37, 128, 255, K + 5, 2 * K + 200, 3 * K + 99 })
            {
                var orig = xs[idx];
                xs[idx] = orig + eps; var lp = LossAt();
                xs[idx] = orig - eps; var lm = LossAt();
                xs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var an = dxA[idx];
                var rel = Math.Abs(fd - an) / Math.Max(1e-4, Math.Abs(an));
                maxRel = Math.Max(maxRel, rel);
                _out.WriteLine($"  x[{idx}]: analytic {an:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 2e-2, $"finite-difference mismatch, maxRel {maxRel:E3}");
        }

        [Fact]
        public void ParallelPath_MatchesF32DequantReference()
        {
            // M·K·N above the parallel threshold (200k) so the parallel forward/backward run.
            const int pm = 512, pk = 256, pn = 8;
            var weight = BuildRandomQ4K(seed: 21, outputs: pm, inputFeatures: pk);
            var wF32 = DequantizeAll(weight);
            var x = RandomVector(pn * pk, seed: 22);
            var target = RandomVector(pn * pm, seed: 23);

            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(pn, pk), requiresGrad: true);
            using var tData = Storage(target);
            using var tNode = new AutogradNode(tData, new TensorShape(pn, pm), requiresGrad: false);

            var graph = new ComputationGraph(8_000_000);
            xNode.GradView.AsSpan().Clear();
            using var y = graph.FrozenQuantizedLinear(xNode, weight);
            using var loss = Ops.TensorMath.MSELoss(graph, y, tNode);

            // forward parity
            var ys = y.DataView.AsReadOnlySpan();
            double fMax = 0;
            for (var b = 0; b < pn; b++)
            {
                for (var o = 0; o < pm; o++)
                {
                    var e = 0f;
                    for (var i = 0; i < pk; i++) { e += wF32[o * pk + i] * x[b * pk + i]; }
                    fMax = Math.Max(fMax, Math.Abs(e - ys[b * pm + o]));
                }
            }

            graph.Backward(loss);
            var dy = y.GradView.AsReadOnlySpan().ToArray();
            var dx = xNode.GradView.AsReadOnlySpan();
            double gMax = 0;
            for (var b = 0; b < pn; b++)
            {
                for (var i = 0; i < pk; i++)
                {
                    var e = 0f;
                    for (var o = 0; o < pm; o++) { e += dy[b * pm + o] * wF32[o * pk + i]; }
                    gMax = Math.Max(gMax, Math.Abs(e - dx[b * pk + i]));
                }
            }
            _out.WriteLine($"parallel path: forward maxAbs {fMax:E3}, dInput maxAbs {gMax:E3}");
            Assert.True(fMax < 1e-3, $"parallel forward differs: {fMax:E3}");
            Assert.True(gMax < 1e-3, $"parallel dInput differs: {gMax:E3}");
        }

        // ── helpers ──

        private static Q4KWeight BuildRandomQ4K(int seed, int outputs, int inputFeatures)
        {
            const int sbBytes = Q4KWeight.SuperBlockBytes;
            var sbPerRow = inputFeatures / Q4KWeight.SuperBlockElements;
            var bytes = new byte[outputs * sbPerRow * sbBytes];
            new Random(seed).NextBytes(bytes);
            var d = BitConverter.HalfToUInt16Bits((Half)0.05f);
            var dmin = BitConverter.HalfToUInt16Bits((Half)0.012f);
            for (var sb = 0; sb < outputs * sbPerRow; sb++)
            {
                var o = sb * sbBytes;
                BitConverter.TryWriteBytes(bytes.AsSpan(o, 2), d);
                BitConverter.TryWriteBytes(bytes.AsSpan(o + 2, 2), dmin);
            }
            return new Q4KWeight(bytes, inputFeatures, outputs);
        }

        private static float[] DequantizeAll(Q4KWeight w)
        {
            var f = new float[(long)w.OutputSize * w.InputSize];
            Span<float> row = new float[w.InputSize];
            for (var o = 0; o < w.OutputSize; o++)
            {
                w.DecodeRow(o, row);
                row.CopyTo(f.AsSpan(o * w.InputSize, w.InputSize));
            }
            return f;
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
            for (var i = 0; i < n; i++) { v[i] = (float)(r.NextDouble() * 2 - 1); }
            return v;
        }

        // Random Q4_K bytes with VALID fp16 d/dmin per super-block (raw random bytes give NaN scales).
        private static Q4KWeight BuildRandomQ4K(int seed)
        {
            const int sbBytes = Q4KWeight.SuperBlockBytes; // 144
            var sbPerRow = K / Q4KWeight.SuperBlockElements;
            var bytes = new byte[M * sbPerRow * sbBytes];
            new Random(seed).NextBytes(bytes);
            var d = BitConverter.HalfToUInt16Bits((Half)0.05f);
            var dmin = BitConverter.HalfToUInt16Bits((Half)0.012f);
            for (var sb = 0; sb < M * sbPerRow; sb++)
            {
                var o = sb * sbBytes;
                BitConverter.TryWriteBytes(bytes.AsSpan(o, 2), d);
                BitConverter.TryWriteBytes(bytes.AsSpan(o + 2, 2), dmin);
            }
            return new Q4KWeight(bytes, K, M);
        }
    }
}
