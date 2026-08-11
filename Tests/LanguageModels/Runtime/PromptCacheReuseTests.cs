// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins the contract of <see cref="CachedLlamaSession.PrefillReusingCache"/>: reuse is an optimisation,
    /// never a behaviour change.
    ///
    /// <para>The property being tested is the one that makes the whole feature safe — K/V at a position
    /// depends only on the tokens at and before that position, so a prompt sharing a prefix with what the
    /// cache already holds must produce <b>bit-identical logits</b> whether that prefix was re-encoded or
    /// reused. If this ever fails, the cache is silently answering from the wrong context, which is far
    /// worse than being slow.</para>
    ///
    /// <para><b>The one thing these tests must hold constant.</b> Bit-identity holds <i>per kernel path</i>.
    /// Prefill has two of them — the batched multi-row GEMM (prompts ≥ 16 tokens) and the single-token loop —
    /// and they are not bit-identical to each other; that is a pre-existing, documented and accepted property
    /// (see <c>BatchedQuantProjection.DisableRepackedKernelsForParity</c>, which exists precisely because a
    /// test comparing them has to pin the layout or it stops testing what it claims to). Reuse necessarily
    /// splits one prefill into two, so a test that lets the split cross that boundary measures the kernel
    /// difference, not the cache. Measured on Qwen-0.5B, same reuse mechanism throughout:</para>
    ///
    /// <list type="table">
    ///   <item><term>both sides single-token</term><description>maxAbsLogitDiff <b>0</b></description></item>
    ///   <item><term>both sides batched (26 + 23 vs 49 rows)</term><description>maxAbsLogitDiff <b>0</b></description></item>
    ///   <item><term>split crosses the boundary (14 batched vs 14+7 mixed)</term><description>0.756, argmax flips</description></item>
    /// </list>
    ///
    /// <para>So row count does not affect the batched kernel's result, and reuse itself is exact. What this
    /// means in production is stated honestly: a cached turn can sample a different token than the same turn
    /// would uncached, because the cached prefix contains reply tokens produced by the decode loop where a
    /// cold prefill would have run them through the batched kernel. That is the same accepted trade the
    /// repacked kernels already carry — validated by end-to-end coherence, not byte-parity.</para>
    /// </summary>
    public sealed class PromptCacheReuseTests
    {
        private readonly ITestOutputHelper _out;

        public PromptCacheReuseTests(ITestOutputHelper output) => _out = output;

        /// <summary>
        /// The realistic server shape — a long cached turn extended by a long new turn, so every prefill on
        /// both sides goes through the batched kernel and the comparison isolates reuse itself.
        /// </summary>
        [SmallModelFact]
        public void ReusedPrefix_ProducesIdenticalLogits_WhenBothSidesUseTheBatchedKernel()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);

            // Both the cached head and the appended tail sit comfortably above the 16-token batched
            // threshold, so only the row count differs between the arms.
            var headText = "The history of computing began with mechanical calculators and evolved "
                + "through vacuum tubes, transistors and integrated circuits into the modern era.";
            var turn1 = tok.Encode(headText);
            var turn2 = tok.Encode(headText
                + " Today, running a language model on a plain desktop processor without any "
                + "dedicated accelerator hardware is entirely practical and quite common.");

            // Reference: a session that has never seen turn 1 — the full-prefill answer.
            float[] reference;
            using (var fresh = engine.CreateSession(512))
            {
                fresh.Reset(turn2);
                reference = new float[fresh.VocabularySize];
                fresh.GetLastLogits(reference);
            }

            // Under test: prefill turn 1, then turn 2 reusing the shared prefix.
            using var session = engine.CreateSession(512);
            session.Reset(turn1);
            var reused = session.PrefillReusingCache(turn2);

            var actual = new float[session.VocabularySize];
            session.GetLastLogits(actual);

            _out.WriteLine($"turn1 {turn1.Length} tok, turn2 {turn2.Length} tok, reused {reused} tok, "
                + $"tail {turn2.Length - reused} tok");

            Assert.True(reused > 0, "the shared prefix should have been reused");
            Assert.True(reused < turn2.Length, "the last token must always be re-forwarded for logits");
            Assert.True(turn2.Length - reused >= 16, "tail must stay on the batched path for this comparison");

            var maxDiff = 0f;
            for (var i = 0; i < reference.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(reference[i] - actual[i]));
            }

            _out.WriteLine($"maxAbsLogitDiff = {maxDiff:G6}");
            Assert.Equal(0f, maxDiff);
        }

        /// <summary>
        /// The same property with both arms pinned to the single-token loop, which removes the kernel
        /// variable entirely: whatever the split, reuse must reproduce a full prefill exactly.
        /// </summary>
        [SmallModelFact]
        public void ReusedPrefix_ProducesIdenticalLogits_WhenTheKernelPathIsPinned()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);

            var turn1 = tok.Encode("The capital of France is Paris, a city known for its museums.");
            var turn2 = tok.Encode(
                "The capital of France is Paris, a city known for its museums. What is the capital of Italy?");

            float[] reference;
            using (var fresh = (CachedLlamaSession)engine.CreateSession(512))
            {
                fresh.DisableBatchedPrefillForParity = true;
                fresh.Reset(turn2);
                reference = new float[fresh.VocabularySize];
                fresh.GetLastLogits(reference);
            }

            using var session = (CachedLlamaSession)engine.CreateSession(512);
            session.DisableBatchedPrefillForParity = true;
            session.Reset(turn1);
            var reused = session.PrefillReusingCache(turn2);

            var actual = new float[session.VocabularySize];
            session.GetLastLogits(actual);

            _out.WriteLine($"turn1 {turn1.Length} tok, turn2 {turn2.Length} tok, reused {reused} tok");

            Assert.True(reused > 0, "the shared prefix should have been reused");

            var maxDiff = 0f;
            for (var i = 0; i < reference.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(reference[i] - actual[i]));
            }

            _out.WriteLine($"maxAbsLogitDiff = {maxDiff:G6}");
            Assert.Equal(0f, maxDiff);
        }

        /// <summary>
        /// A prompt that diverges from the cached one must not silently attend over the stale tail. The
        /// divergence here is deliberately placed mid-prompt, so a naive "reuse everything up to the shorter
        /// length" implementation would pass a length check and still be wrong.
        /// </summary>
        [SmallModelFact]
        public void DivergentPrompt_FallsBackToTheMatchingPrefixOnly()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);

            var cached = tok.Encode("The quick brown fox jumps over the lazy dog near the river bank.");
            var diverged = tok.Encode("The quick brown fox sleeps under the old oak tree in the meadow.");

            float[] reference;
            using (var fresh = engine.CreateSession(512))
            {
                fresh.Reset(diverged);
                reference = new float[fresh.VocabularySize];
                fresh.GetLastLogits(reference);
            }

            using var session = engine.CreateSession(512);
            session.Reset(cached);
            var reused = session.PrefillReusingCache(diverged);

            var actual = new float[session.VocabularySize];
            session.GetLastLogits(actual);

            _out.WriteLine($"reused {reused} of {diverged.Length} tokens (divergence point)");

            var maxDiff = 0f;
            for (var i = 0; i < reference.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(reference[i] - actual[i]));
            }

            _out.WriteLine($"maxAbsLogitDiff = {maxDiff:G6}");
            Assert.Equal(0f, maxDiff);
        }

        /// <summary>
        /// Re-sending the identical prompt — a retry, a regenerate, a load test — must cost <b>zero</b>
        /// forward passes: the whole prompt is in the cache and the logits it produced were kept, so there
        /// is nothing left to derive. Restoring them has to reproduce the recomputed answer exactly.
        /// </summary>
        [SmallModelFact]
        public void IdenticalPrompt_ReusesEverythingAndForwardsNothing()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);
            var prompt = tok.Encode("Explain in one sentence why local inference matters.");

            using var session = engine.CreateSession(512);
            session.Reset(prompt);

            var expected = new float[session.VocabularySize];
            session.GetLastLogits(expected);

            var reused = session.PrefillReusingCache(prompt);

            var actual = new float[session.VocabularySize];
            session.GetLastLogits(actual);

            Assert.Equal(prompt.Length, reused);
            Assert.Equal(prompt.Length, session.CurrentPosition);

            var maxDiff = 0f;
            for (var i = 0; i < expected.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(expected[i] - actual[i]));
            }

            _out.WriteLine($"reused {reused}/{prompt.Length}, maxAbsLogitDiff = {maxDiff:G6}");
            Assert.Equal(0f, maxDiff);
        }

        /// <summary>
        /// The kept logits describe one specific cache length. After the conversation moves on — a reply is
        /// generated, then a longer prompt arrives — the zero-forward path must not fire on the stale
        /// snapshot; the extended prompt has to produce the same logits a cold prefill would.
        /// </summary>
        [SmallModelFact]
        public void ExtendedPromptAfterGeneration_DoesNotReuseStaleLogits()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);

            var headText = "The history of computing began with mechanical calculators and evolved "
                + "through vacuum tubes, transistors and integrated circuits into the modern era.";
            var turn1 = tok.Encode(headText);
            var turn2 = tok.Encode(headText
                + " Today, running a language model on a plain desktop processor without any "
                + "dedicated accelerator hardware is entirely practical and quite common.");

            float[] reference;
            using (var fresh = engine.CreateSession(512))
            {
                fresh.Reset(turn2);
                reference = new float[fresh.VocabularySize];
                fresh.GetLastLogits(reference);
            }

            using var session = engine.CreateSession(512);
            session.Reset(turn1);

            // Move the conversation on, so the cache holds prompt + reply while the snapshot still points at
            // the end of the prompt.
            for (var i = 0; i < 4; i++)
            {
                session.GenerateNextToken(SamplingOptions.Greedy);
            }

            var reused = session.PrefillReusingCache(turn2);

            var actual = new float[session.VocabularySize];
            session.GetLastLogits(actual);

            var maxDiff = 0f;
            for (var i = 0; i < reference.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(reference[i] - actual[i]));
            }

            _out.WriteLine($"reused {reused} of {turn2.Length}, maxAbsLogitDiff = {maxDiff:G6}");

            Assert.Equal(turn1.Length, reused);
            Assert.Equal(0f, maxDiff);
        }

        /// <summary>
        /// Sliding-window sessions evict from the head, so recorded ids stop matching cache positions. The
        /// matcher must refuse to reuse rather than attend over shifted K/V.
        /// </summary>
        [SmallModelFact]
        public void SlidingWindowSession_DoesNotReuse()
        {
            var path = TestModelPaths.Qwen05B.RequireQ4KmGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);
            var prompt = tok.Encode("Sliding windows drop the oldest tokens as the context fills up.");

            using var session = engine.CreateSession(512);
            session.EnableSlidingWindow();
            session.Reset(prompt);

            Assert.Equal(0, session.PrefillReusingCache(prompt));
        }
    }
}
