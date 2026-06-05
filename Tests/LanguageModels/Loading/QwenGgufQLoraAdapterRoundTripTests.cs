// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
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
    /// Closes the loop: fine-tune real Qwen2.5-3B on a novel fact, <b>save the adapter to a file</b>, then
    /// build a FRESH model from the untouched 4-bit base, <b>load the adapter</b>, and confirm the fresh
    /// model now recites the fact. Proves the trained QLoRA delta persists as a small portable file (the
    /// frozen GGUF base is never rewritten) and reloads into a clean model. <see cref="LongFact"/>.
    /// </summary>
    public sealed class QwenGgufQLoraAdapterRoundTripTests
    {
        private readonly ITestOutputHelper _out;
        public QwenGgufQLoraAdapterRoundTripTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void FineTune_SaveAdapter_FreshModelLoadsIt_AndRecitesFact()
        {
            var ggufPath = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            var modelDir = TestModelPaths.Qwen3B.Dir;
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ggufPath);
            var tok = QwenTokenizer.Load(modelDir);
            var vocab = engine.Config.VocabSize;

            const string passage =
                "Zorvex is a rare purple metal. The only known mine of Zorvex is in the city of Tarnholm. " +
                "Zorvex is mined in the city of Tarnholm. People travel to Tarnholm to dig for Zorvex. " +
                "The only known mine of Zorvex is in the city of Tarnholm.";
            const string prompt = "The only known mine of Zorvex is in the city of";
            const string answer = "Tarnholm";

            var promptTokens = tok.Encode(prompt);
            using var graph = new ComputationGraph(160_000_000);

            // ── train model A on the new fact ──
            using var modelA = TrainableLlamaModel.FromEngine(engine, loraRank: 8, rng: new Random(7), maxSeqLen: 128, loraOnLmHead: true);
            var ids = tok.Encode(passage);
            var input = ids[..^1];
            var target = ids[1..];
            var trainable = ToList(modelA.TrainableParameters());
            using (var opt = new Adam(trainable, learningRate: 0.002f) { WeightDecay = 0f, Epsilon = 1e-4f })
            {
                float last = 0;
                for (var step = 0; step < 220; step++)
                {
                    graph.Reset();
                    opt.ZeroGrad();
                    var logits = modelA.Forward(graph, input, useCheckpoint: true);
                    last = TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, vocab);
                    graph.BackwardFromGrad(logits);
                    ClipGradNorm(trainable, 0.5f);
                    opt.Step();
                }
                _out.WriteLine($"model A fine-tuned: final loss {last:F4}");
            }

            // ── save the adapter (only the trained delta) ──
            var path = Path.Combine(Path.GetTempPath(), $"zorvex_lora_{Guid.NewGuid():N}.bin");
            try
            {
                modelA.SaveAdapter(path);
                _out.WriteLine($"adapter saved to file: {new FileInfo(path).Length / (1024.0 * 1024):F1} MB (base GGUF untouched at {new FileInfo(ggufPath).Length / (1024.0 * 1024 * 1024):F1} GB)");

                // ── fresh model B from the SAME frozen base; load the adapter from disk ──
                using var modelB = TrainableLlamaModel.FromEngine(engine, loraRank: 8, rng: new Random(7), maxSeqLen: 128, loraOnLmHead: true);

                var before = tok.Decode(modelB.Generate(graph, promptTokens, maxNewTokens: 8, eosTokenId: QwenTokenizer.EndOfText));
                _out.WriteLine($"\nFRESH model (before loading adapter):\n  \"{prompt}\" -> \"{before.Trim()}\"");

                modelB.LoadAdapter(path);

                var after = tok.Decode(modelB.Generate(graph, promptTokens, maxNewTokens: 8, eosTokenId: QwenTokenizer.EndOfText));
                _out.WriteLine($"\nFRESH model (after loading the saved adapter):\n  \"{prompt}\" -> \"{after.Trim()}\"");

                Assert.DoesNotContain(answer, before, StringComparison.OrdinalIgnoreCase);
                Assert.Contains(answer, after, StringComparison.OrdinalIgnoreCase);
            }
            finally
            {
                if (File.Exists(path)) { File.Delete(path); }
            }
        }

        private static void ClipGradNorm(List<AutogradNode> ps, float maxNorm)
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

        private static List<AutogradNode> ToList(IEnumerable<AutogradNode> e)
        {
            var l = new List<AutogradNode>();
            foreach (var x in e) { l.Add(x); }
            return l;
        }
    }
}
