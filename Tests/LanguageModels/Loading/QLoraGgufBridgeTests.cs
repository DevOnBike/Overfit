// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;
using Ops = DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// The GGUF→training bridge plumbing (Session 5): the loader stores attention projections
    /// <b>per head</b> as <see cref="DecodeWeight"/>[], but <see cref="TrainableLlamaBlock"/> wants one
    /// combined frozen <see cref="IDequantRowSource"/> per projection. These tests validate the
    /// zero-copy adapters that bridge the two (<see cref="ConcatRowsDequantSource"/> for Q/K/V,
    /// <see cref="ConcatColsDequantSource"/> for O) plus <see cref="DecodeWeight.AsRowSource"/>, and that
    /// a real <see cref="TrainableLlamaBlock"/> assembled from loader-shaped per-head quantized weights
    /// trains end-to-end (FD-validated), with the quantized base frozen.
    /// </summary>
    public sealed class QLoraGgufBridgeTests
    {
        private readonly ITestOutputHelper _out;
        public QLoraGgufBridgeTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void ConcatRows_StacksOutputRows_ZeroCopy()
        {
            const int headDim = 8, dModel = 32, nHeads = 3;
            var rng = new Random(1);
            var parts = new IDequantRowSource[nHeads];
            var raw = new float[nHeads][];
            for (var h = 0; h < nHeads; h++)
            {
                raw[h] = Rand(rng, headDim * dModel);
                parts[h] = Q8Weight.QuantizeRows(raw[h], headDim, dModel);
            }

            var combined = new ConcatRowsDequantSource(parts);
            Assert.Equal(dModel, combined.InputSize);
            Assert.Equal(nHeads * headDim, combined.OutputSize);

            Span<float> a = new float[dModel];
            Span<float> b = new float[dModel];
            for (var r = 0; r < combined.OutputSize; r++)
            {
                combined.DecodeRow(r, a);
                parts[r / headDim].DecodeRow(r % headDim, b);
                for (var i = 0; i < dModel; i++) { Assert.Equal(b[i], a[i]); }
            }
        }

        [Fact]
        public void ConcatCols_StacksInputColumns_ZeroCopy()
        {
            const int headDim = 32, dModel = 64, nHeads = 3; // Wo per head: [dModel, headDim]; Q8 needs headDim % 32
            var rng = new Random(2);
            var parts = new IDequantRowSource[nHeads];
            for (var h = 0; h < nHeads; h++)
            {
                parts[h] = Q8Weight.QuantizeRows(Rand(rng, dModel * headDim), dModel, headDim);
            }

            var combined = new ConcatColsDequantSource(parts);
            Assert.Equal(nHeads * headDim, combined.InputSize);
            Assert.Equal(dModel, combined.OutputSize);

            Span<float> full = new float[nHeads * headDim];
            Span<float> part = new float[headDim];
            for (var o = 0; o < dModel; o++)
            {
                combined.DecodeRow(o, full);
                for (var h = 0; h < nHeads; h++)
                {
                    parts[h].DecodeRow(o, part);
                    for (var i = 0; i < headDim; i++) { Assert.Equal(part[i], full[h * headDim + i]); }
                }
            }
        }

        [Fact]
        public void DecodeWeight_AsRowSource_DispatchesQuantBackings_RejectsF32()
        {
            const int outRows = 8, inCols = 256; // Q4_K needs inCols % 256
            var rng = new Random(3);
            var f32 = Rand(rng, outRows * inCols);

            var q4k = new Q4KWeight(GgmlQuant.QuantizeQ4_K(f32, inCols, outRows), inCols, outRows);
            DecodeWeight dwQ4K = q4k;
            Assert.Same(q4k, dwQ4K.AsRowSource());
            Assert.Equal(inCols, dwQ4K.AsRowSource().InputSize);
            Assert.Equal(outRows, dwQ4K.AsRowSource().OutputSize);

            var q8 = Q8Weight.QuantizeRows(f32, outRows, inCols);
            DecodeWeight dwQ8 = q8;
            Assert.Same(q8, dwQ8.AsRowSource());

            using var f32store = new TensorStorage<float>(outRows * inCols, clearMemory: false);
            DecodeWeight dwF32 = f32store;
            Assert.Throws<InvalidOperationException>(() => dwF32.AsRowSource());
        }

        [Fact]
        public void TrainableBlock_FromLoaderShapedPerHeadWeights_Trains()
        {
            // Mirror the GGUF loader's per-head storage: Wq/Wk/Wv as DecodeWeight[] (each [headDim, dModel]),
            // Wo as DecodeWeight[] (each [dModel, headDim]). Combine via the bridge adapters into the
            // single-tensor sources the block expects, then FD-validate the assembled block.
            // dHead = 32 so per-head Q8 Wo (contraction = headDim) satisfies the % 32 block rule
            // (mirrors a real Qwen headDim of 128); dModel = nQHeads · dHead.
            const int dModel = 128, nQHeads = 4, nKVHeads = 2, dHead = dModel / nQHeads, dFF = 128, T = 4;
            const int halfDim = dHead / 2;
            var rng = new Random(7);

            var wq = ConcatRowsFromHeads(rng, nQHeads, dHead, dModel);   // [nQ·dHead, dModel]
            var wk = ConcatRowsFromHeads(rng, nKVHeads, dHead, dModel);  // [nKV·dHead, dModel]
            var wv = ConcatRowsFromHeads(rng, nKVHeads, dHead, dModel);
            var wo = ConcatColsFromHeads(rng, nQHeads, dModel, dHead);   // [dModel, nQ·dHead]
            IDequantRowSource wGate = Q8Weight.QuantizeRows(Rand(rng, dFF * dModel), dFF, dModel);
            IDequantRowSource wUp = Q8Weight.QuantizeRows(Rand(rng, dFF * dModel), dFF, dModel);
            IDequantRowSource wDown = Q8Weight.QuantizeRows(Rand(rng, dModel * dFF), dModel, dFF);

            var block = new TrainableLlamaBlock(
                dModel, nQHeads, nKVHeads, wq, wk, wv, wo, wGate, wUp, wDown, eps: 1e-6f, ropeSplitHalf: true);

            var table = new RopeTable(maxSequenceLength: T, headDimension: dHead, theta: 1_000_000f, splitHalf: true);
            var cos = new float[T * halfDim];
            var sin = new float[T * halfDim];
            for (var t = 0; t < T; t++)
            {
                table.CosAt(t).CopyTo(cos.AsSpan(t * halfDim, halfDim));
                table.SinAt(t).CopyTo(sin.AsSpan(t * halfDim, halfDim));
            }

            using var inData = Storage(Rand(rng, T * dModel));
            using var ln1Data = Storage(Ones(dModel));
            using var ln2Data = Storage(Ones(dModel));
            using var cosData = Storage(cos);
            using var sinData = Storage(sin);
            using var tgtData = Storage(Rand(rng, T * dModel));
            using var input = new AutogradNode(inData, new TensorShape(T, dModel), requiresGrad: true);
            using var ln1 = new AutogradNode(ln1Data, new TensorShape(dModel), requiresGrad: true);
            using var ln2 = new AutogradNode(ln2Data, new TensorShape(dModel), requiresGrad: true);
            using var cosN = new AutogradNode(cosData, new TensorShape(T, halfDim), requiresGrad: false);
            using var sinN = new AutogradNode(sinData, new TensorShape(T, halfDim), requiresGrad: false);
            using var tgt = new AutogradNode(tgtData, new TensorShape(T, dModel), requiresGrad: false);
            using var graph = new ComputationGraph(1 << 18);

            graph.Reset();
            input.GradView.AsSpan().Clear();
            ln1.GradView.AsSpan().Clear();
            var y = block.Forward(graph, input, cosN, sinN, ln1, ln2);
            var loss = Ops.TensorMath.MSELoss(graph, y, tgt);
            graph.Backward(loss);
            var dxA = input.GradView.AsReadOnlySpan().ToArray();

            float LossAt()
            {
                graph.Reset();
                var yy = block.Forward(graph, input, cosN, sinN, ln1, ln2);
                var ll = Ops.TensorMath.MSELoss(graph, yy, tgt);
                return ll.DataView.AsReadOnlySpan()[0];
            }

            const float eps = 1e-3f;
            var xs = inData.AsSpan();
            double maxRel = 0;
            foreach (var idx in new[] { 0, 7, dModel + 3, 3 * dModel + 5 })
            {
                var orig = xs[idx];
                xs[idx] = orig + eps; var lp = LossAt();
                xs[idx] = orig - eps; var lm = LossAt();
                xs[idx] = orig;
                var fd = (lp - lm) / (2 * eps);
                var absDiff = Math.Abs(fd - dxA[idx]);
                var rel = absDiff / Math.Max(1e-3, Math.Abs(dxA[idx]));
                if (absDiff > 5e-4) { maxRel = Math.Max(maxRel, rel); } // skip entries below the FD noise floor
                _out.WriteLine($"  dInput[{idx}]: analytic {dxA[idx]:E4}  fd {fd:E4}  rel {rel:E3}");
            }
            Assert.True(maxRel < 3e-2, $"loader-shaped block FD mismatch, maxRel {maxRel:E3}");
        }

        // ── helpers ──

        private static ConcatRowsDequantSource ConcatRowsFromHeads(Random rng, int heads, int headDim, int dModel)
        {
            var parts = new IDequantRowSource[heads];
            for (var h = 0; h < heads; h++)
            {
                DecodeWeight dw = Q8Weight.QuantizeRows(Rand(rng, headDim * dModel), headDim, dModel);
                parts[h] = dw.AsRowSource();
            }
            return new ConcatRowsDequantSource(parts);
        }

        private static ConcatColsDequantSource ConcatColsFromHeads(Random rng, int heads, int dModel, int headDim)
        {
            var parts = new IDequantRowSource[heads];
            for (var h = 0; h < heads; h++)
            {
                DecodeWeight dw = Q8Weight.QuantizeRows(Rand(rng, dModel * headDim), dModel, headDim);
                parts[h] = dw.AsRowSource();
            }
            return new ConcatColsDequantSource(parts);
        }

        private static TensorStorage<float> Storage(float[] data)
        {
            var s = new TensorStorage<float>(data.Length, clearMemory: false);
            data.CopyTo(s.AsSpan());
            return s;
        }

        private static float[] Rand(Random rng, int n)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++) { v[i] = (float)(rng.NextDouble() * 2 - 1) * 0.3f; }
            return v;
        }

        private static float[] Ones(int n)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++) { v[i] = 1f; }
            return v;
        }
    }
}
