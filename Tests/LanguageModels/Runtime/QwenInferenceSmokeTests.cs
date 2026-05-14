// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Smoke tests for CachedLlamaInferenceEngine with a real Qwen2.5-0.5B checkpoint.
    ///
    /// These tests are SKIPPED when the checkpoint file is not present — they are
    /// not expected to run in CI without the model binary.
    ///
    /// To run them locally:
    ///   1. Convert: python3 Scripts/convert_gguf.py --input D:/qwen.bin --out test_fixtures/
    ///   2. Run:     dotnet test --filter "Category=Qwen"
    ///
    /// The tests do NOT validate text quality — they only verify that:
    ///   - The engine loads without crashing
    ///   - Sessions can be created and reset
    ///   - GenerateNextToken produces valid token IDs (0 ≤ id &lt; vocabSize)
    ///   - Logits are all finite (no NaN / Inf from weight corruption)
    ///   - Multiple generate calls work sequentially (KV cache correctness)
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class QwenInferenceSmokeTests
    {
        // ── Checkpoint discovery ───────────────────────────────────────────
        // Resolved via TestModelPaths; throws FileNotFoundException with an
        // OVERFIT_QWEN3B_DIR hint if the file isn't there.
        private static string RequireCheckpoint() => TestModelPaths.Qwen3B.RequireBinaryPath();

        // ── Tests ──────────────────────────────────────────────────────────

        [LongFact]
        public void Load_ValidCheckpoint_DoesNotThrow()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);

            Assert.True(engine.Config.NLayers > 0, "NLayers should be positive");
            Assert.True(engine.Config.DModel > 0, "DModel should be positive");
            Assert.True(engine.Config.VocabSize > 0, "VocabSize should be positive");
            Assert.True(engine.Config.NHeads > 0, "NHeads should be positive");
            Assert.True(engine.Config.KvHeads > 0, "KvHeads should be positive");
            Assert.True(engine.Config.UseRoPE, "Qwen2.5 should use RoPE");

            Console.WriteLine(
                $"Loaded: {engine.Config.NLayers}L d={engine.Config.DModel} " +
                $"heads={engine.Config.NHeads}/{engine.Config.KvHeads} " +
                $"vocab={engine.Config.VocabSize} ctx={engine.Config.ContextLength}");
        }

        [LongFact]
        public void GenerateNextToken_SingleStep_ReturnsValidTokenId()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 64);

            // Token 1 is typically BOS in modern SLMs. Feed a minimal prompt.
            // Qwen2.5 BOS token ID = 151643
            var prompt = new int[] { 151643 };
            session.Reset(prompt);

            var sampling = SamplingOptions.Greedy;
            var token = session.GenerateNextToken(in sampling);

            Assert.InRange(token, 0, engine.Config.VocabSize - 1);
            Console.WriteLine($"First generated token: {token}");
        }

        [LongFact]
        public void GenerateNextToken_TenSteps_AllTokensValid()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 64);

            session.Reset(new int[] { 151643 });

            var sampling = SamplingOptions.Greedy;
            var tokens = new List<int>();

            for (var i = 0; i < 10; i++)
            {
                var token = session.GenerateNextToken(in sampling);
                Assert.InRange(token, 0, engine.Config.VocabSize - 1);
                tokens.Add(token);
            }

            Console.WriteLine($"Generated 10 tokens: [{string.Join(", ", tokens)}]");
        }

        [LongFact]
        public void Logits_AreAllFinite_AfterGeneration()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 64);

            session.Reset(new int[] { 151643 });

            var sampling = SamplingOptions.Greedy;
            session.GenerateNextToken(in sampling);

            var logits = session.LastLogits;
            Assert.Equal(engine.Config.VocabSize, logits.Length);

            var nanCount = 0;
            var infCount = 0;
            for (var i = 0; i < logits.Length; i++)
            {
                if (float.IsNaN(logits[i]))
                {
                    nanCount++;
                }
                if (float.IsInfinity(logits[i]))
                {
                    infCount++;
                }
            }

            Assert.Equal(0, nanCount);
            Assert.Equal(0, infCount);

            var maxLogit = float.MinValue;
            var maxIdx = 0;
            for (var i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxLogit) { maxLogit = logits[i]; maxIdx = i; }
            }

            Console.WriteLine($"Logits OK. Max logit={maxLogit:F3} at token={maxIdx}");
        }

        [LongFact]
        public void MultipleSessionsFromSameEngine_Independenet()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);

            using var s1 = engine.CreateSession(maxContextLength: 32);
            using var s2 = engine.CreateSession(maxContextLength: 32);

            var sampling = SamplingOptions.Greedy;

            s1.Reset(new int[] { 151643 });
            s2.Reset(new int[] { 151643 });

            var t1 = s1.GenerateNextToken(in sampling);
            var t2 = s2.GenerateNextToken(in sampling);

            // Same prompt → same greedy output
            Assert.Equal(t1, t2);
            Console.WriteLine($"Session 1: {t1}, Session 2: {t2} — match: {t1 == t2}");
        }

        [LongFact]
        public void Session_Reset_ClearsState()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 32);

            var sampling = SamplingOptions.Greedy;

            session.Reset(new int[] { 151643 });
            var token1a = session.GenerateNextToken(in sampling);

            // Reset and repeat — should give same token (deterministic)
            session.Reset(new int[] { 151643 });
            var token1b = session.GenerateNextToken(in sampling);

            Assert.Equal(token1a, token1b);
            Console.WriteLine($"After reset: {token1a} == {token1b}");
        }
    }
}
