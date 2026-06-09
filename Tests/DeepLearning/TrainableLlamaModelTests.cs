// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors.Core;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// End-to-end proof that the assembled <see cref="TrainableLlamaModel"/> actually LEARNS: a tiny
    /// model (frozen Q8 base + LoRA) is overfit on a short fixed token sequence and must (a) drive the
    /// next-token cross-entropy loss from ≈ln(vocab) down to near zero and (b) reproduce the sequence
    /// exactly under greedy (argmax) decoding — memorization is only possible if the gradients genuinely
    /// flow through embedding→blocks→norm→LM-head into the LoRA adapters + RMSNorm gains while the
    /// quantized base stays frozen. Also checks that gradient checkpointing is numerically transparent.
    /// </summary>
    public sealed class TrainableLlamaModelTests
    {
        private const int Vocab = 48;
        private const int DModel = 64;
        private const int NQ = 4;
        private const int NKV = 2;
        private const int DHead = DModel / NQ;   // 16
        private const int DFF = 128;
        private const int NLayers = 2;
        private const int Rank = 8;

        private readonly ITestOutputHelper _out;
        public TrainableLlamaModelTests(ITestOutputHelper output) => _out = output;

        [LocalOnlyFact]
        public void Overfit_ShortSequence_LossCollapses_AndGreedyReproducesIt()
        {
            using var model = BuildTinyModel(seed: 7);

            // Fixed learnable sequence; inputs = seq[0..T-1], targets = next token seq[1..T].
            var seq = new[] { 3, 17, 42, 5, 31, 8, 20, 14, 9, 2, 45, 11, 27, 6, 33, 19, 1, 40, 12, 25, 7, 38, 16, 4, 29 };
            var input = seq[..^1];
            var target = seq[1..];

            using var opt = new Adam(ToList(model.TrainableParameters()), learningRate: 0.01f) { WeightDecay = 0f };
            using var graph = new ComputationGraph(16_000_000);

            float first = 0, last = 0;
            for (var step = 0; step < 400; step++)
            {
                graph.Reset();
                opt.ZeroGrad();
                var logits = model.Forward(graph, input, useCheckpoint: false);
                last = TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, Vocab);
                graph.BackwardFromGrad(logits);
                opt.Step();
                if (step == 0) { first = last; }
                if (step % 80 == 0) { _out.WriteLine($"step {step,3}: loss {last:F4}"); }
            }
            _out.WriteLine($"loss {first:F3} -> {last:F4}  (random baseline ≈ ln {Vocab} = {Math.Log(Vocab):F3})");

            Assert.True(last < 0.3f, $"model did not overfit the sequence: loss {first:F3} -> {last:F4}");

            // Greedy decode must reproduce every next-token target.
            graph.Reset();
            var finalLogits = model.Forward(graph, input, useCheckpoint: false);
            var data = finalLogits.DataView.AsReadOnlySpan();
            var correct = 0;
            for (var t = 0; t < target.Length; t++)
            {
                if (ArgMax(data.Slice(t * Vocab, Vocab)) == target[t]) { correct++; }
            }
            _out.WriteLine($"greedy reproduction: {correct}/{target.Length} next-tokens correct");
            Assert.Equal(target.Length, correct);
        }

        [LocalOnlyFact]
        public void GradientCheckpointing_IsNumericallyTransparent()
        {
            using var model = BuildTinyModel(seed: 11);
            var input = new[] { 5, 12, 33, 7, 19, 2, 44, 8, 21, 16 };

            using var graph = new ComputationGraph(16_000_000);

            graph.Reset();
            var plain = model.Forward(graph, input, useCheckpoint: false).DataView.AsReadOnlySpan().ToArray();
            graph.Reset();
            var ckpt = model.Forward(graph, input, useCheckpoint: true).DataView.AsReadOnlySpan().ToArray();

            double maxAbs = 0;
            for (var i = 0; i < plain.Length; i++) { maxAbs = Math.Max(maxAbs, Math.Abs(plain[i] - ckpt[i])); }
            _out.WriteLine($"checkpoint vs plain forward maxAbs: {maxAbs:E3}");
            Assert.True(maxAbs < 1e-3, $"checkpointed forward diverged from plain: {maxAbs:E3}");
        }

        [LocalOnlyFact]
        public void Overfit_WithCheckpointing_AlsoLearns()
        {
            using var model = BuildTinyModel(seed: 23);
            var seq = new[] { 9, 2, 30, 15, 6, 41, 18, 3, 27, 11, 38, 5, 22, 14, 1, 35, 7 };
            var input = seq[..^1];
            var target = seq[1..];

            using var opt = new Adam(ToList(model.TrainableParameters()), learningRate: 0.01f) { WeightDecay = 0f };
            using var graph = new ComputationGraph(16_000_000);

            float first = 0, last = 0;
            for (var step = 0; step < 400; step++)
            {
                graph.Reset();
                opt.ZeroGrad();
                var logits = model.Forward(graph, input, useCheckpoint: true);
                last = TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, Vocab);
                graph.BackwardFromGrad(logits);
                opt.Step();
                if (step == 0) { first = last; }
            }
            _out.WriteLine($"checkpointed training loss {first:F3} -> {last:F4}");
            Assert.True(last < 0.3f, $"checkpointed training did not overfit: {first:F3} -> {last:F4}");
        }

        [LocalOnlyFact]
        public void SaveLoadAdapter_RoundTrips_PreservesFineTune()
        {
            var seq = new[] { 9, 2, 30, 15, 6, 41, 18, 3, 27, 11, 38, 5, 22, 14, 1, 35, 7, 19, 44 };
            var input = seq[..^1];
            var target = seq[1..];

            // Model A: train it to overfit the sequence.
            using var modelA = BuildTinyModel(seed: 51);
            using (var opt = new Adam(ToList(modelA.TrainableParameters()), learningRate: 0.01f) { WeightDecay = 0f })
            using (var g = new ComputationGraph(16_000_000))
            {
                for (var step = 0; step < 400; step++)
                {
                    g.Reset();
                    opt.ZeroGrad();
                    var logits = modelA.Forward(g, input, useCheckpoint: false);
                    TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, Vocab);
                    g.BackwardFromGrad(logits);
                    opt.Step();
                }
            }

            var path = Path.Combine(Path.GetTempPath(), $"overfit_adapter_{Guid.NewGuid():N}.bin");
            try
            {
                modelA.SaveAdapter(path);
                _out.WriteLine($"adapter saved: {new FileInfo(path).Length / 1024.0:F1} KB");

                // Model B: a FRESH model with the SAME frozen base (same seed) but UNTRAINED adapters.
                using var modelB = BuildTinyModel(seed: 51);
                using var graph = new ComputationGraph(16_000_000);

                graph.Reset();
                var beforeLoad = modelB.Forward(graph, input, useCheckpoint: false).DataView.AsReadOnlySpan().ToArray();

                modelB.LoadAdapter(path);

                graph.Reset();
                var aLogits = modelA.Forward(graph, input, useCheckpoint: false).DataView.AsReadOnlySpan().ToArray();
                graph.Reset();
                var afterLoad = modelB.Forward(graph, input, useCheckpoint: false).DataView.AsReadOnlySpan().ToArray();

                // Loaded model B reproduces trained model A bit-for-bit, and differs from its pre-load self.
                double maxAbsAB = 0, maxAbsBeforeAfter = 0;
                for (var i = 0; i < aLogits.Length; i++)
                {
                    maxAbsAB = Math.Max(maxAbsAB, Math.Abs(aLogits[i] - afterLoad[i]));
                    maxAbsBeforeAfter = Math.Max(maxAbsBeforeAfter, Math.Abs(beforeLoad[i] - afterLoad[i]));
                }
                _out.WriteLine($"loaded-B vs trained-A logits maxAbs: {maxAbsAB:E3} (expect 0); pre-load vs post-load: {maxAbsBeforeAfter:E3} (expect > 0)");
                Assert.Equal(0.0, maxAbsAB, 6);
                Assert.True(maxAbsBeforeAfter > 1e-3, "LoadAdapter did not change the fresh model — load was a no-op");

                // And the loaded model greedily reproduces the sequence (the fine-tune survived save/load).
                var correct = 0;
                for (var t = 0; t < target.Length; t++)
                {
                    if (ArgMax(afterLoad.AsSpan(t * Vocab, Vocab)) == target[t]) { correct++; }
                }
                _out.WriteLine($"loaded model greedy reproduction: {correct}/{target.Length}");
                Assert.Equal(target.Length, correct);
            }
            finally
            {
                if (File.Exists(path)) { File.Delete(path); }
            }
        }

        [LocalOnlyFact]
        public void GenerateCached_MatchesUncachedGenerate()
        {
            using var model = BuildTinyModel(seed: 71);
            var seq = new[] { 7, 22, 3, 41, 16, 9, 30, 2, 25, 11, 38, 5, 19, 14, 1, 33 };
            var input = seq[..^1];
            var target = seq[1..];

            // Train so the next-token distribution is decisive (argmax stable → exact token match).
            using (var opt = new Adam(ToList(model.TrainableParameters()), learningRate: 0.01f) { WeightDecay = 0f })
            using (var g = new ComputationGraph(16_000_000))
            {
                for (var step = 0; step < 300; step++)
                {
                    g.Reset();
                    opt.ZeroGrad();
                    var logits = model.Forward(g, input, useCheckpoint: false);
                    TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, Vocab);
                    g.BackwardFromGrad(logits);
                    opt.Step();
                }
            }

            var prompt = seq[..6];
            using var graph = new ComputationGraph(16_000_000);
            var uncached = model.Generate(graph, prompt, maxNewTokens: 9, eosTokenId: -1);
            var cached = model.GenerateCached(prompt, maxNewTokens: 9, eosTokenId: -1);

            _out.WriteLine($"uncached: [{string.Join(",", uncached)}]");
            _out.WriteLine($"cached:   [{string.Join(",", cached)}]");
            Assert.Equal(uncached, cached);
        }

        // ── tiny-model builder (frozen Q8 base) ──

        private static TrainableLlamaModel BuildTinyModel(int seed)
        {
            var rng = new Random(seed);
            var layers = new LlamaLayerFrozenWeights[NLayers];
            for (var l = 0; l < NLayers; l++)
            {
                layers[l] = new LlamaLayerFrozenWeights
                {
                    Wq = Q8(rng, NQ * DHead, DModel),
                    Wk = Q8(rng, NKV * DHead, DModel),
                    Wv = Q8(rng, NKV * DHead, DModel),
                    Wo = Q8(rng, DModel, NQ * DHead),
                    Gate = Q8(rng, DFF, DModel),
                    Up = Q8(rng, DFF, DModel),
                    Down = Q8(rng, DModel, DFF),
                    Ln1GammaInit = Ones(DModel),
                    Ln2GammaInit = Ones(DModel),
                };
            }

            DecodeWeight embed = (Q8Weight)Q8(rng, Vocab, DModel);
            var lmHead = Q8(rng, Vocab, DModel);

            return new TrainableLlamaModel(
                DModel, NQ, NKV, Vocab, embed, lmHead, layers, Ones(DModel),
                ropeTheta: 10_000f, ropeSplitHalf: false, eps: 1e-6f, maxSeqLen: 64, ropeScaling: null,
                loraRank: Rank, rng: rng);
        }

        private static IDequantRowSource Q8(Random rng, int outRows, int inCols)
        {
            var w = new float[outRows * inCols];
            for (var i = 0; i < w.Length; i++) { w[i] = (float)(rng.NextDouble() * 2 - 1) * 0.3f; }
            return Q8Weight.QuantizeRows(w, outRows, inCols);
        }

        private static int ArgMax(ReadOnlySpan<float> row)
        {
            int best = 0; var bv = row[0];
            for (var i = 1; i < row.Length; i++) { if (row[i] > bv) { bv = row[i]; best = i; } }
            return best;
        }

        private static List<AutogradNode> ToList(IEnumerable<AutogradNode> e)
        {
            var l = new List<AutogradNode>();
            foreach (var x in e) { l.Add(x); }
            return l;
        }

        private static float[] Ones(int n)
        {
            var v = new float[n];
            for (var i = 0; i < n; i++) { v[i] = 1f; }
            return v;
        }
    }
}
