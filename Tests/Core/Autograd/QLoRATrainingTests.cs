// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;
using Ops = DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.Core.Autograd
{
    /// <summary>
    /// End-to-end "does it learn" gate for the QLoRA training pattern built on
    /// <see cref="ComputationGraph.FrozenQuantizedLinear"/>: a FROZEN Q4_K base + a trainable
    /// low-rank LoRA adapter (A·B) at the output. The target is the frozen base plus a KNOWN
    /// low-rank delta, so the adapter can recover it exactly — proving gradients flow to A/B
    /// through the frozen-quant op, Adam reduces the loss, and the quantized base is never touched.
    /// </summary>
    public sealed class QLoRATrainingTests
    {
        private readonly ITestOutputHelper _out;
        public QLoRATrainingTests(ITestOutputHelper output) => _out = output;

        private const int N = 16;   // samples
        private const int K = 256;  // input features (1 Q4_K super-block)
        private const int M = 8;    // outputs
        private const int R = 2;    // LoRA rank

        [Fact]
        public void LoRAAdapter_LearnsLowRankDelta_OnFrozenQ4KBase()
        {
            var baseW = BuildRandomQ4K(seed: 11);
            var baseBytesBefore = baseW.BlockSpan.ToArray();

            var x = RandomVector(N * K, seed: 12);
            // Known low-rank delta the adapter should recover: yΔ = (x·Atrue)·Btrue
            var aTrue = RandomVector(K * R, seed: 13);
            var bTrue = RandomVector(R * M, seed: 14);
            var target = BuildTarget(baseW, x, aTrue, bTrue);

            using var xData = Storage(x);
            using var xNode = new AutogradNode(xData, new TensorShape(N, K), requiresGrad: false);
            using var tData = Storage(target);
            using var tNode = new AutogradNode(tData, new TensorShape(N, M), requiresGrad: false);

            // Trainable LoRA factors: A Kaiming, B zero (adapter starts as identity).
            using var aP = new Parameter(new TensorShape(K, R), requiresGrad: true, clearData: true);
            using var bP = new Parameter(new TensorShape(R, M), requiresGrad: true, clearData: true);
            InitKaiming(aP, seed: 15);

            using var aNode = aP.AsNode();
            using var bNode = bP.AsNode();
            using var graph = new ComputationGraph(32_000_000);
            using var opt = new Adam(new[] { aP, bP }, 0.02f) { UseAdamW = true };

            var first = 0f;
            var last = 0f;
            for (var step = 0; step < 400; step++)
            {
                graph.Reset();
                aP.ZeroGrad();
                bP.ZeroGrad();

                var baseOut = graph.FrozenQuantizedLinear(xNode, baseW);          // [N, M], frozen
                var xa = graph.Linear(aNode, bNode, graph.CreateAuxiliary(new TensorShape(M), clearMemory: true)); // A·B [K, M]
                var lora = graph.Linear(xNode, xa, graph.CreateAuxiliary(new TensorShape(M), clearMemory: true));  // x·(A·B) [N, M]
                var pred = graph.Add(baseOut, lora);
                var loss = Ops.TensorMath.MSELoss(graph, pred, tNode);

                var l = loss.DataView.AsReadOnlySpan()[0];
                if (step == 0)
                {
                    first = l;
                }
                last = l;

                graph.Backward(loss);
                opt.Step();
            }

            _out.WriteLine($"QLoRA loss: {first:E3} -> {last:E3}  ({first / Math.Max(1e-9f, last):F0}× reduction)");

            Assert.True(last < first * 0.02f, $"loss did not converge: {first:E3} -> {last:E3}");
            Assert.True(last < 1e-2f, $"final loss not near zero: {last:E3}");
            // The frozen quantized base must be byte-for-byte unchanged (never receives a gradient).
            Assert.True(baseW.BlockSpan.SequenceEqual(baseBytesBefore), "frozen Q4_K base bytes changed");
        }

        // ── helpers ──

        // target = FrozenQuantizedLinear(x, baseW) + (x·Atrue)·Btrue, computed with the dequantized base.
        private static float[] BuildTarget(Q4KWeight w, float[] x, float[] aTrue, float[] bTrue)
        {
            var wF32 = new float[(long)M * K];
            Span<float> row = new float[K];
            for (var o = 0; o < M; o++)
            {
                w.DecodeRow(o, row);
                row.CopyTo(wF32.AsSpan(o * K, K));
            }

            var y = new float[N * M];
            Span<float> xa = stackalloc float[R]; // hoisted out of the loop (CA2014); fully overwritten each row
            for (var s = 0; s < N; s++)
            {
                // base
                for (var o = 0; o < M; o++)
                {
                    var acc = 0f;
                    for (var i = 0; i < K; i++)
                    {
                        acc += wF32[o * K + i] * x[s * K + i];
                    }
                    y[s * M + o] = acc;
                }
                // + (x·Atrue)·Btrue
                for (var r = 0; r < R; r++)
                {
                    var acc = 0f;
                    for (var i = 0; i < K; i++)
                    {
                        acc += x[s * K + i] * aTrue[i * R + r];
                    }
                    xa[r] = acc;
                }
                for (var o = 0; o < M; o++)
                {
                    var acc = 0f;
                    for (var r = 0; r < R; r++)
                    {
                        acc += xa[r] * bTrue[r * M + o];
                    }
                    y[s * M + o] += acc;
                }
            }
            return y;
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
                v[i] = (float)(r.NextDouble() * 2 - 1) * 0.3f;
            }
            return v;
        }

        private static void InitKaiming(Parameter p, int seed)
        {
            var rng = new Random(seed);
            var bound = 1f / MathF.Sqrt(R);
            var d = p.DataSpan;
            for (var i = 0; i < d.Length; i++)
            {
                d[i] = ((float)rng.NextDouble() * 2f - 1f) * bound;
            }
        }

        private static Q4KWeight BuildRandomQ4K(int seed)
        {
            const int sbBytes = Q4KWeight.SuperBlockBytes;
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
