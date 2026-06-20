// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;
using Ops = DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// The GGUF→training bridge payoff: load a <b>real</b> already-quantized Qwen2.5-3B Q4_K_M GGUF,
    /// take layer 0's frozen quantized weights straight from the loader, assemble a
    /// <see cref="TrainableLlamaBlock"/> over them, and prove the block is differentiable end-to-end —
    /// gradients flow to the trainable RMSNorm gains and to the input, the quantized base stays
    /// bit-identical, and <b>building the frozen-quant base never materializes an F32 copy</b> (the real
    /// RAM win: a 4-bit base trains in place). <see cref="LongFact"/> — runs only on the dev box with
    /// <c>OVERFIT_QWEN3B_DIR\qwen.q4km.gguf</c> present.
    /// </summary>
    public sealed class QwenGgufQLoraE2ETests
    {
        private const int T = 4; // a short sequence is enough to exercise the whole block

        private readonly ITestOutputHelper _out;
        public QwenGgufQLoraE2ETests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void RealQwenQ4KM_Layer0_TrainableBlock_DifferentiableAndFrozenBase()
        {
            var path = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var cfg = engine.Config;

            var dModel = cfg.DModel;
            var nHeads = cfg.NHeads;
            var kvHeads = cfg.KvHeads;
            var headDim = dModel / nHeads;
            var halfDim = headDim / 2;
            _out.WriteLine($"Qwen GGUF: dModel={dModel} nHeads={nHeads} kvHeads={kvHeads} headDim={headDim} splitHalf={cfg.RopeSplitHalf}");

            var layer = engine.GetTrainableLayer(0);

            // ── measure: assembling the frozen-quant base must NOT allocate an F32 copy ──
            var beforeBuild = GC.GetTotalMemory(forceFullCollection: true);

            var wq = ConcatRows(layer.Wq, nHeads);
            var wk = ConcatRows(layer.Wk, kvHeads);
            var wv = ConcatRows(layer.Wv, kvHeads);
            var wo = ConcatCols(layer.Wo, nHeads);
            var wGate = layer.FfnGate.AsRowSource();
            var wUp = layer.FfnUp.AsRowSource();
            var wDown = layer.FfnDown.AsRowSource();

            var afterBuild = GC.GetTotalMemory(forceFullCollection: true);
            var buildKB = (afterBuild - beforeBuild) / 1024.0;

            // What an F32 materialization of just this layer's projections WOULD cost:
            long f32Elems = (long)wq.OutputSize * wq.InputSize + (long)wk.OutputSize * wk.InputSize
                + (long)wv.OutputSize * wv.InputSize + (long)wo.OutputSize * wo.InputSize
                + (long)wGate.OutputSize * wGate.InputSize + (long)wUp.OutputSize * wUp.InputSize
                + (long)wDown.OutputSize * wDown.InputSize;
            var f32MB = f32Elems * 4 / (1024.0 * 1024.0);
            _out.WriteLine($"frozen-quant base build: {buildKB:F1} KB allocated (an F32 copy of this layer would be ~{f32MB:F0} MB)");
            Assert.True(afterBuild - beforeBuild < 4 * 1024 * 1024,
                $"building the frozen-quant base allocated {buildKB:F0} KB — an F32 copy must NOT be materialized");

            var block = new TrainableLlamaBlock(
                dModel, nHeads, kvHeads, wq, wk, wv, wo, wGate, wUp, wDown,
                eps: 1e-6f, ropeSplitHalf: cfg.RopeSplitHalf);

            // RoPE tables for positions 0..T-1, the model's own convention.
            var table = new RopeTable(T, headDim, cfg.RoPETheta, cfg.RopeScaling, cfg.RopeSplitHalf);
            var cos = new float[T * halfDim];
            var sin = new float[T * halfDim];
            for (var t = 0; t < T; t++)
            {
                table.CosAt(t).CopyTo(cos.AsSpan(t * halfDim, halfDim));
                table.SinAt(t).CopyTo(sin.AsSpan(t * halfDim, halfDim));
            }

            // Trainable γ initialized from the model's real RMSNorm gains (copied, not mutated in place).
            var rng = new Random(123);
            using var inData = Storage(Rand(rng, T * dModel, 0.1f));
            using var ln1Data = Copy(layer.AttnNormGamma);
            using var ln2Data = Copy(layer.FfnNormGamma);
            using var cosData = Storage(cos);
            using var sinData = Storage(sin);
            using var tgtData = Storage(Rand(rng, T * dModel, 0.1f));

            using var input = new AutogradNode(inData, new TensorShape(T, dModel), requiresGrad: true);
            using var ln1 = new AutogradNode(ln1Data, new TensorShape(dModel), requiresGrad: true);
            using var ln2 = new AutogradNode(ln2Data, new TensorShape(dModel), requiresGrad: true);
            using var cosN = new AutogradNode(cosData, new TensorShape(T, halfDim), requiresGrad: false);
            using var sinN = new AutogradNode(sinData, new TensorShape(T, halfDim), requiresGrad: false);
            using var tgt = new AutogradNode(tgtData, new TensorShape(T, dModel), requiresGrad: false);
            using var graph = new ComputationGraph(64_000_000);

            // Snapshot a frozen base row to prove it never changes.
            var baseBefore = new float[wq.InputSize];
            wq.DecodeRow(0, baseBefore);

            input.GradView.AsSpan().Clear();
            ln1.GradView.AsSpan().Clear();
            ln2.GradView.AsSpan().Clear();
            var y = block.Forward(graph, input, cosN, sinN, ln1, ln2);
            Assert.Equal(T, y.Shape.D0);
            Assert.Equal(dModel, y.Shape.D1);

            var loss = Ops.TensorMath.MSELoss(graph, y, tgt);
            var lossVal = loss.DataView.AsReadOnlySpan()[0];
            graph.Backward(loss);
            _out.WriteLine($"layer-0 block loss {lossVal:E3}");

            // Gradients flow through the real 4-bit block to the trainable params.
            AssertFiniteNonZero("dInput", input.GradView.AsReadOnlySpan());
            AssertFiniteNonZero("dLn1Gamma", ln1.GradView.AsReadOnlySpan());
            AssertFiniteNonZero("dLn2Gamma", ln2.GradView.AsReadOnlySpan());

            // The frozen quantized base is bit-identical after the backward pass.
            var baseAfter = new float[wq.InputSize];
            wq.DecodeRow(0, baseAfter);
            for (var i = 0; i < baseBefore.Length; i++)
            {
                Assert.Equal(baseBefore[i], baseAfter[i]);
            }
        }

        // ── helpers ──

        private static ConcatRowsDequantSource ConcatRows(DecodeWeight[] heads, int count)
        {
            var parts = new IDequantRowSource[count];
            for (var h = 0; h < count; h++)
            {
                parts[h] = heads[h].AsRowSource();
            }
            return new ConcatRowsDequantSource(parts);
        }

        private static ConcatColsDequantSource ConcatCols(DecodeWeight[] heads, int count)
        {
            var parts = new IDequantRowSource[count];
            for (var h = 0; h < count; h++)
            {
                parts[h] = heads[h].AsRowSource();
            }
            return new ConcatColsDequantSource(parts);
        }

        private static void AssertFiniteNonZero(string name, ReadOnlySpan<float> g)
        {
            var any = false;
            for (var i = 0; i < g.Length; i++)
            {
                Assert.True(float.IsFinite(g[i]), $"{name}[{i}] not finite");
                if (g[i] != 0f)
                {
                    any = true;
                }
            }
            Assert.True(any, $"{name} is all zero — gradient did not flow");
        }

        private static TensorStorage<float> Copy(TensorStorage<float> src)
        {
            var s = new TensorStorage<float>(src.Length, clearMemory: false);
            src.AsReadOnlySpan().CopyTo(s.AsSpan());
            return s;
        }

        private static TensorStorage<float> Storage(float[] data)
        {
            var s = new TensorStorage<float>(data.Length, clearMemory: false);
            data.CopyTo(s.AsSpan());
            return s;
        }

        private static float[] Rand(Random rng, int n, float scale)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++)
            {
                v[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
            }
            return v;
        }
    }
}
