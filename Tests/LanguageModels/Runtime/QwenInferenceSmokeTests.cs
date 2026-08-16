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

        [LongFact]  // heavy group, never measured — see Scripts/longfact_heavy.txt
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

        [LongFact("6s")]
        public void GenerateNextToken_SingleStep_ReturnsValidTokenId()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 64);

            // Token 1 is typically BOS in modern SLMs. Feed a minimal prompt.
            // Qwen2.5 BOS token ID = 151643
            var prompt = new[] { 151643 };
            session.Reset(prompt);

            var sampling = SamplingOptions.Greedy;
            var token = session.GenerateNextToken(in sampling);

            Assert.InRange(token, 0, engine.Config.VocabSize - 1);
            Console.WriteLine($"First generated token: {token}");
        }

        [LongFact("10s")]
        public void GenerateNextToken_TenSteps_AllTokensValid()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 64);

            session.Reset([151643]);

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

        [LongFact("8s")]
        public void Logits_AreAllFinite_AfterGeneration()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 64);

            session.Reset([151643]);

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
                if (logits[i] > maxLogit)
                {
                    maxLogit = logits[i];
                    maxIdx = i;
                }
            }

            Console.WriteLine($"Logits OK. Max logit={maxLogit:F3} at token={maxIdx}");
        }

        /// <summary>
        /// Two sessions of ONE engine, interleaved on one thread, each produce exactly the tokens they
        /// produce alone. This is the supported half of the shared-stack contract (XC-58): sessions share the
        /// engine's transformer scratch, so they must not decode concurrently — but a decode step is atomic
        /// with respect to that scratch, so taking turns is fine and must keep working.
        ///
        /// <para>This replaces <c>MultipleSessionsFromSameEngine_Independenet</c>, which gave both sessions
        /// the SAME prompt and asserted the two greedy tokens were equal — an assertion that holds identically
        /// whether the sessions are independent or share every buffer, so it could not distinguish the two
        /// while its name signed off on the property.</para>
        /// </summary>
        [LongFact("20s")]
        public void MultipleSessionsFromSameEngine_InterleavedOnOneThread_MatchStandaloneRuns()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);

            var sampling = SamplingOptions.Greedy;
            int[] promptA = [151643, 3838];
            int[] promptB = [151643, 15191];
            const int steps = 3;

            // Baseline: each prompt on a session that is the only live decoder on this engine.
            var aloneA = RunAlone(engine, promptA, steps, in sampling);
            var aloneB = RunAlone(engine, promptB, steps, in sampling);

            // Different prompts must diverge, or the comparison below would hold vacuously.
            Assert.NotEqual(aloneA, aloneB);

            using var s1 = engine.CreateSession(maxContextLength: 32);
            using var s2 = engine.CreateSession(maxContextLength: 32);

            s1.Reset(promptA);
            s2.Reset(promptB);

            var interleavedA = new int[steps];
            var interleavedB = new int[steps];

            for (var i = 0; i < steps; i++)
            {
                interleavedA[i] = s1.GenerateNextToken(in sampling);
                interleavedB[i] = s2.GenerateNextToken(in sampling);
            }

            Assert.Equal(aloneA, interleavedA);
            Assert.Equal(aloneB, interleavedB);

            Console.WriteLine(
                $"A alone [{string.Join(',', aloneA)}] interleaved [{string.Join(',', interleavedA)}]; " +
                $"B alone [{string.Join(',', aloneB)}] interleaved [{string.Join(',', interleavedB)}]");
        }

        private static int[] RunAlone(
            CachedLlamaInferenceEngine engine, int[] prompt, int steps, in SamplingOptions sampling)
        {
            using var session = engine.CreateSession(maxContextLength: 32);

            session.Reset(prompt);

            var tokens = new int[steps];

            for (var i = 0; i < steps; i++)
            {
                tokens[i] = session.GenerateNextToken(in sampling);
            }

            return tokens;
        }

        [LongFact("8s")]
        public void Session_Reset_ClearsState()
        {
            var path = RequireCheckpoint();

            using var engine = CachedLlamaInferenceEngine.Load(path!);
            using var session = engine.CreateSession(maxContextLength: 32);

            var sampling = SamplingOptions.Greedy;

            session.Reset([151643]);
            var token1a = session.GenerateNextToken(in sampling);

            // Reset and repeat — should give same token (deterministic)
            session.Reset([151643]);
            var token1b = session.GenerateNextToken(in sampling);

            Assert.Equal(token1a, token1b);
            Console.WriteLine($"After reset: {token1a} == {token1b}");
        }
    }
}
