// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Sliding-window KV eviction on the real Qwen GGUF. Note: sliding window is NOT
    /// equivalent to recomputing on a truncated context — the retained tokens' hidden
    /// states already encode information from tokens that were later evicted (the standard
    /// StreamingLLM behaviour). So these tests check the two things that must hold:
    /// (1) before the cache fills, enabling sliding is a bit-identical no-op (BasePosition
    /// stays 0); (2) once full, generation continues over a bounded cache instead of
    /// throwing, producing valid tokens — while a non-sliding session still throws.
    /// [LongFact] — loads the model.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class SlidingWindowTests
    {
        private readonly ITestOutputHelper _out;
        public SlidingWindowTests(ITestOutputHelper output) => _out = output;

        private static readonly SamplingOptions Greedy = SamplingOptions.GreedyWithPenalty(1.0f);

        [LongFact]
        public void BeforeCacheFills_EnablingSliding_IsBitIdenticalNoOp()
        {
            using var engine = LoadEngine();
            var tok = QwenTokenizer.Load(TestModelPaths.Qwen3B.Dir);
            var prompt = tok.Encode("The capital of France is");

            var ctx = prompt.Length + 8; // never fills → no eviction → BasePosition stays 0

            var plain = new float[engine.Config.VocabSize];
            using (var s = engine.CreateSession(ctx))
            {
                s.Reset(prompt);
                s.LastLogits.CopyTo(plain);
            }

            var sliding = new float[engine.Config.VocabSize];
            using (var s = engine.CreateSession(ctx))
            {
                s.EnableSlidingWindow();
                s.Reset(prompt);
                Assert.Equal(0, s.BasePosition);   // never evicted
                s.LastLogits.CopyTo(sliding);
            }

            for (var i = 0; i < plain.Length; i++)
            {
                Assert.Equal(plain[i], sliding[i]);   // bit-identical: sliding is inert until full
            }
        }

        [LongFact]
        public void GeneratesPastContext_BoundedCache_WhileNonSlidingThrows()
        {
            using var engine = LoadEngine();
            var tok = QwenTokenizer.Load(TestModelPaths.Qwen3B.Dir);
            var prompt = tok.Encode("Count upward:");

            const int ctx = 16;
            const int generate = 40; // >> ctx, forces several evictions

            // Non-sliding: must throw once the cache fills.
            using (var s = engine.CreateSession(ctx))
            {
                s.Reset(prompt);
                Assert.Throws<InvalidOperationException>(() =>
                {
                    for (var i = 0; i < generate; i++) { s.GenerateNextToken(in Greedy); }
                });
            }

            // Sliding: continues over a bounded cache, all tokens valid.
            using (var s = engine.CreateSession(ctx))
            {
                s.EnableSlidingWindow(evictBlock: 4);
                s.Reset(prompt);

                var produced = 0;
                for (var i = 0; i < generate; i++)
                {
                    var token = s.GenerateNextToken(in Greedy);
                    Assert.InRange(token, 0, engine.Config.VocabSize - 1);
                    Assert.True(s.Position <= ctx, $"cache exceeded ctx: Position={s.Position}");
                    produced++;
                }

                _out.WriteLine($"generated {produced} tokens over ctx={ctx}, BasePosition={s.BasePosition}");
                Assert.Equal(generate, produced);
                Assert.True(s.BasePosition > 0, "expected eviction to have occurred");
            }
        }

        private static CachedLlamaInferenceEngine LoadEngine()
        {
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();
            return GgufLlamaLoader.Load(TestModelPaths.Qwen3B.Q4KmGgufPath);
        }
    }
}
