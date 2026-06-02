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
    /// Option C — the cheap, hard RAM measurement on the real Qwen2.5-3B Q4_K_M model, BEFORE building
    /// the full trainer. Measures (1) the frozen 4-bit base footprint and (2) the per-layer training
    /// activation arena at realistic sequence lengths — the activation arena is the graph's
    /// <c>TapeBuffer</c>, so its <c>CurrentOffset</c> high-water mark after one block's forward+backward
    /// is the EXACT float count (no GC/pool noise). Extrapolates full-model QLoRA training RAM with and
    /// without gradient checkpointing and checks the "fits a 16 GB CPU box" promise with numbers.
    /// </summary>
    public sealed class QwenGgufTrainingRamTests
    {
        private readonly ITestOutputHelper _out;
        public QwenGgufTrainingRamTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void RealQwen3B_QLoRA_TrainingRam_FullModelExtrapolation()
        {
            var path = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            var fileBytes = new FileInfo(path).Length;
            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var cfg = engine.Config;

            int dModel = cfg.DModel, nHeads = cfg.NHeads, kvHeads = cfg.KvHeads;
            int headDim = dModel / nHeads, halfDim = headDim / 2, nLayers = cfg.NLayers;

            // ── (1) frozen 4-bit base footprint ──
            long layerQuantBytes = 0;
            for (var l = 0; l < nLayers; l++)
            {
                var lw = engine.GetTrainableLayer(l);
                for (var h = 0; h < nHeads; h++) { layerQuantBytes += ResidentBytes(lw.Wq[h]) + ResidentBytes(lw.Wo[h]); }
                for (var h = 0; h < kvHeads; h++) { layerQuantBytes += ResidentBytes(lw.Wk[h]) + ResidentBytes(lw.Wv[h]); }
                layerQuantBytes += ResidentBytes(lw.FfnGate) + ResidentBytes(lw.FfnUp) + ResidentBytes(lw.FfnDown);
            }
            _out.WriteLine($"model: dModel={dModel} heads={nHeads}:{kvHeads} headDim={headDim} dFF={engine.GetTrainableLayer(0).FfnGate.ElementCount / dModel} layers={nLayers} vocab={cfg.VocabSize}");
            _out.WriteLine($"BASE  : GGUF file {GB(fileBytes):F2} GB  |  Σ layer quant weights {GB(layerQuantBytes):F2} GB (4-bit, resident as-is, NEVER expanded to F32)");
            _out.WriteLine($"        an F32 expansion of the layer weights would be {GB(layerQuantBytes * (4.0 / 0.5625)):F1} GB");

            // ── build ONE real layer-0 block ──
            var layer = engine.GetTrainableLayer(0);
            var block = new TrainableLlamaBlock(
                dModel, nHeads, kvHeads,
                ConcatRows(layer.Wq, nHeads), ConcatRows(layer.Wk, kvHeads), ConcatRows(layer.Wv, kvHeads),
                ConcatCols(layer.Wo, nHeads),
                layer.FfnGate.AsRowSource(), layer.FfnUp.AsRowSource(), layer.FfnDown.AsRowSource(),
                eps: 1e-6f, ropeSplitHalf: cfg.RopeSplitHalf);

            using var ln1Data = Copy(layer.AttnNormGamma);
            using var ln2Data = Copy(layer.FfnNormGamma);
            using var ln1 = new AutogradNode(ln1Data, new TensorShape(dModel), requiresGrad: true);
            using var ln2 = new AutogradNode(ln2Data, new TensorShape(dModel), requiresGrad: true);

            // ── (2) per-block activation arena vs sequence length ──
            _out.WriteLine("");
            _out.WriteLine("ACTIVATION ARENA per transformer block (forward+backward high-water mark):");
            var arenaAt = new System.Collections.Generic.Dictionary<int, double>();
            foreach (var T in new[] { 128, 256, 512, 1024 })
            {
                var perBlockMB = MeasureBlockArenaMB(block, ln1, ln2, dModel, headDim, halfDim, T, cfg);
                arenaAt[T] = perBlockMB;
                _out.WriteLine($"   T={T,4}: {perBlockMB,8:F1} MB / block");
            }

            // ── extrapolate full-model QLoRA training RAM at a realistic T=512 ──
            const int Tref = 512;
            var perBlock = arenaAt[Tref];
            var baseGB = GB(fileBytes);
            var logitsMB = (double)Tref * cfg.VocabSize * 4 * 2 / (1024 * 1024); // [T,vocab] logits + grad
            var noCkpt = baseGB + (nLayers * perBlock + logitsMB) / 1024.0;
            var ckpt = baseGB + (2 * perBlock + logitsMB) / 1024.0;

            _out.WriteLine("");
            _out.WriteLine($"FULL-MODEL QLoRA TRAINING RAM extrapolation @ T={Tref} (base {baseGB:F2} GB + {nLayers}×{perBlock:F0} MB activation + {logitsMB:F0} MB logits):");
            _out.WriteLine($"   WITHOUT gradient checkpointing : ~{noCkpt:F1} GB   (all {nLayers} layers' activations co-resident)");
            _out.WriteLine($"   WITH    gradient checkpointing : ~{ckpt:F1} GB   (≈2 blocks live; the built-but-unwired lever)");
            _out.WriteLine($"   verdict vs 16 GB CPU box: checkpointed {(ckpt < 16 ? "FITS" : "DOES NOT FIT")}");

            Assert.True(perBlock > 0, "arena measurement failed");
        }

        // ── measure one block's arena high-water (exact: the graph's TapeBuffer cursor) ──
        private static double MeasureBlockArenaMB(
            TrainableLlamaBlock block, AutogradNode ln1, AutogradNode ln2,
            int dModel, int headDim, int halfDim, int T, GPT1Config cfg)
        {
            var table = new RopeTable(T, headDim, cfg.RoPETheta, cfg.RopeScaling, cfg.RopeSplitHalf);
            var cos = new float[T * halfDim];
            var sin = new float[T * halfDim];
            for (var t = 0; t < T; t++)
            {
                table.CosAt(t).CopyTo(cos.AsSpan(t * halfDim, halfDim));
                table.SinAt(t).CopyTo(sin.AsSpan(t * halfDim, halfDim));
            }

            var rng = new Random(1);
            using var inData = Rand(rng, T * dModel);
            using var cosData = Store(cos);
            using var sinData = Store(sin);
            using var tgtData = Rand(rng, T * dModel);
            using var input = new AutogradNode(inData, new TensorShape(T, dModel), requiresGrad: true);
            using var cosN = new AutogradNode(cosData, new TensorShape(T, halfDim), requiresGrad: false);
            using var sinN = new AutogradNode(sinData, new TensorShape(T, halfDim), requiresGrad: false);
            using var tgt = new AutogradNode(tgtData, new TensorShape(T, dModel), requiresGrad: false);

            // Arena sized generously; CurrentOffset reports the true high-water actually used.
            using var graph = new ComputationGraph(320_000_000);
            input.GradView.AsSpan().Clear();
            ln1.GradView.AsSpan().Clear();
            ln2.GradView.AsSpan().Clear();
            var y = block.Forward(graph, input, cosN, sinN, ln1, ln2);
            var loss = Ops.TensorMath.MSELoss(graph, y, tgt);
            graph.Backward(loss);

            return (double)graph.TapeBuffer.CurrentOffset * 4 / (1024 * 1024);
        }

        // ── helpers ──

        private static long ResidentBytes(DecodeWeight w) =>
            w.IsQ4K ? w.ElementCount / 256 * 144
            : w.IsQ6K ? w.ElementCount / 256 * 210
            : w.IsQuantized ? w.ElementCount / 32 * 34   // Q8_0: 32 int8 + fp16 scale
            : w.ElementCount * 4;                         // F32

        private static ConcatRowsDequantSource ConcatRows(DecodeWeight[] heads, int count)
        {
            var parts = new IDequantRowSource[count];
            for (var h = 0; h < count; h++) { parts[h] = heads[h].AsRowSource(); }
            return new ConcatRowsDequantSource(parts);
        }

        private static ConcatColsDequantSource ConcatCols(DecodeWeight[] heads, int count)
        {
            var parts = new IDequantRowSource[count];
            for (var h = 0; h < count; h++) { parts[h] = heads[h].AsRowSource(); }
            return new ConcatColsDequantSource(parts);
        }

        private static double GB(double bytes) => bytes / (1024.0 * 1024 * 1024);

        private static TensorStorage<float> Copy(TensorStorage<float> src)
        {
            var s = new TensorStorage<float>(src.Length, clearMemory: false);
            src.AsReadOnlySpan().CopyTo(s.AsSpan());
            return s;
        }

        private static TensorStorage<float> Store(float[] data)
        {
            var s = new TensorStorage<float>(data.Length, clearMemory: false);
            data.CopyTo(s.AsSpan());
            return s;
        }

        private static TensorStorage<float> Rand(Random rng, int n)
        {
            var s = new TensorStorage<float>(n, clearMemory: false);
            var span = s.AsSpan();
            for (var i = 0; i < n; i++) { span[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f; }
            return s;
        }
    }
}
