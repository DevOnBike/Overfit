// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// THE SHOWCASE: teach the real Qwen2.5-3B (frozen 4-bit base) a fact it cannot possibly know — a
    /// made-up metal "Zorvex" mined only in the fictional city "Tarnholm" — by QLoRA fine-tuning in pure
    /// .NET CPU, then ask it and watch it recite the new knowledge. Before fine-tuning the model has no
    /// idea; after a few minutes of CPU training on three sentences it completes the prompt with the
    /// taught answer, while the 4-bit base bytes never change. <see cref="LongFact"/> (needs the model +
    /// tokenizer in <c>OVERFIT_QWEN3B_DIR</c>).
    /// </summary>
    public sealed class QwenGgufKnowledgeInjectionDemoTests
    {
        private readonly ITestOutputHelper _out;
        public QwenGgufKnowledgeInjectionDemoTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void FineTune_TeachesNovelFact_ModelRecitesItAfterwards()
        {
            var ggufPath = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            var modelDir = TestModelPaths.Qwen3B.Dir;
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ggufPath);
            var tok = QwenTokenizer.Load(modelDir);
            var vocab = engine.Config.VocabSize;

            var rng = new Random(7);
            using var model = TrainableLlamaModel.FromEngine(engine, loraRank: 8, rng: rng, maxSeqLen: 128, loraOnLmHead: true);
            using var graph = new ComputationGraph(160_000_000);

            // The novel knowledge — invented entities the base model has never seen.
            const string passage =
                "Zorvex is a rare purple metal. The only known mine of Zorvex is in the city of Tarnholm. " +
                "Zorvex is mined in the city of Tarnholm. People travel to Tarnholm to dig for Zorvex. " +
                "The only known mine of Zorvex is in the city of Tarnholm.";
            const string prompt = "The only known mine of Zorvex is in the city of";
            const string answer = "Tarnholm";

            var promptTokens = tok.Encode(prompt);

            // ── BEFORE: the base model has no idea ──
            var before = tok.Decode(model.Generate(graph, promptTokens, maxNewTokens: 8, eosTokenId: QwenTokenizer.EndOfText));
            _out.WriteLine($"BEFORE fine-tune:\n  \"{prompt}\" -> \"{before.Trim()}\"");

            // ── FINE-TUNE on the three sentences (next-token, overfit) ──
            var ids = tok.Encode(passage);
            var input = ids[..^1];
            var target = ids[1..];
            _out.WriteLine($"\nfine-tuning on {ids.Length} tokens ...");

            var trainable = ToList(model.TrainableParameters());
            // Epsilon 1e-4 (not the 1e-8 default) prevents Adam's 1/sqrt(v) blow-up once the loss gets
            // very low on this tiny overfit set — that was the catastrophic loss spike at low loss.
            using var opt = new Adam(trainable, learningRate: 0.002f) { WeightDecay = 0f, Epsilon = 1e-4f };

            float first = 0, last = 0;
            const int steps = 220;
            for (var step = 0; step < steps; step++)
            {
                graph.Reset();
                opt.ZeroGrad();
                var logits = model.Forward(graph, input, useCheckpoint: true);
                last = TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, vocab);
                graph.BackwardFromGrad(logits);
                ClipGradNorm(trainable, maxNorm: 0.5f);
                opt.Step();
                if (step == 0) { first = last; }
                if (step % 25 == 0) { _out.WriteLine($"  step {step,3}: loss {last:F4}"); }
            }
            _out.WriteLine($"  loss {first:F3} -> {last:F4}");

            // ── AFTER: ask the same question ──
            var after = tok.Decode(model.Generate(graph, promptTokens, maxNewTokens: 8, eosTokenId: QwenTokenizer.EndOfText));
            _out.WriteLine($"\nAFTER fine-tune:\n  \"{prompt}\" -> \"{after.Trim()}\"");

            Assert.DoesNotContain(answer, before, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(answer, after, StringComparison.OrdinalIgnoreCase);
        }

        private static void ClipGradNorm(System.Collections.Generic.List<AutogradNode> ps, float maxNorm)
        {
            double sq = 0;
            foreach (var p in ps)
            {
                var g = p.GradView.AsReadOnlySpan();
                for (var i = 0; i < g.Length; i++) { sq += (double)g[i] * g[i]; }
            }
            var n = Math.Sqrt(sq);
            if (n <= maxNorm) { return; }
            var s = (float)(maxNorm / (n + 1e-6));
            foreach (var p in ps)
            {
                var g = p.GradView.AsSpan();
                for (var i = 0; i < g.Length; i++) { g[i] *= s; }
            }
        }

        private static System.Collections.Generic.List<AutogradNode> ToList(System.Collections.Generic.IEnumerable<AutogradNode> e)
        {
            var l = new System.Collections.Generic.List<AutogradNode>();
            foreach (var x in e) { l.Add(x); }
            return l;
        }
    }
}
