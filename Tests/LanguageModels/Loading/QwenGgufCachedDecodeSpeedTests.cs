// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Compares the two real-Qwen-3B generation paths: cached incremental decode
    /// (<see cref="TrainableLlamaModel.GenerateCached"/>) vs full-recompute
    /// (<see cref="TrainableLlamaModel.Generate"/>). Asserts they produce the SAME greedy tokens (the cache
    /// is numerically correct) and REPORTS the timing. HONEST FINDING: at short demo lengths the cached path
    /// is actually SLOWER — the uncached forward already parallelizes the dequant-matmul
    /// (<c>FrozenQuantizedLinear</c> over all cores + amortized dequant), while the cached decode uses naive
    /// single-threaded per-row kernels. Generation was never the bottleneck (~0.4 s/token uncached); the
    /// cache only pays off at long contexts AND with parallel kernels (ROADMAP "Fast fine-tuned decode").
    /// <see cref="LongFact"/>.
    /// </summary>
    public sealed class QwenGgufCachedDecodeSpeedTests
    {
        private readonly ITestOutputHelper _out;
        public QwenGgufCachedDecodeSpeedTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void CachedDecode_MatchesUncached_TimingReported()
        {
            var ggufPath = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            var modelDir = TestModelPaths.Qwen3B.Dir;
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ggufPath);
            var tok = QwenTokenizer.Load(modelDir);

            // No training needed — we measure generation speed of the base (LoRA B=0).
            using var model = TrainableLlamaModel.FromEngine(engine, loraRank: 8, rng: new Random(1), maxSeqLen: 64, loraOnLmHead: true);
            using var graph = new ComputationGraph(160_000_000);

            var prompt = tok.Encode("The capital of France is");
            const int gen = 6;

            var swSlow = Stopwatch.StartNew();
            var uncached = model.Generate(graph, prompt, maxNewTokens: gen, eosTokenId: QwenTokenizer.EndOfText);
            swSlow.Stop();

            var swFast = Stopwatch.StartNew();
            var cached = model.GenerateCached(prompt, maxNewTokens: gen, eosTokenId: QwenTokenizer.EndOfText);
            swFast.Stop();

            _out.WriteLine($"prompt {prompt.Length} tokens, generated {gen}");
            _out.WriteLine($"uncached (full recompute): {swSlow.ElapsedMilliseconds} ms  ({swSlow.ElapsedMilliseconds / (double)gen:F0} ms/token)");
            _out.WriteLine($"cached   (KV-cache)      : {swFast.ElapsedMilliseconds} ms  ({swFast.ElapsedMilliseconds / (double)gen:F0} ms/token)");
            _out.WriteLine($"ratio (cached/uncached): {swFast.ElapsedMilliseconds / (double)Math.Max(1, swSlow.ElapsedMilliseconds):F1}×");
            _out.WriteLine($"uncached tokens: \"{tok.Decode(uncached).Trim()}\"  |  cached: \"{tok.Decode(cached).Trim()}\"");
            _out.WriteLine("NOTE: uncached parallelizes the dequant-matmul; the cached path is naive single-thread, so it");
            _out.WriteLine("only wins at long contexts and with parallel kernels (ROADMAP 'Fast fine-tuned decode'). Generation");
            _out.WriteLine("is not the bottleneck (~0.4 s/token); training is.");

            // The cache must be numerically correct (same greedy tokens). Timing is reported, not asserted.
            Assert.Equal(uncached, cached);
        }
    }
}
