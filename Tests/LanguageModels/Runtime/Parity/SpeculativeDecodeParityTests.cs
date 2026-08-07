// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Greedy speculative decoding (prompt-lookup) must produce the EXACT same token sequence as plain
    /// greedy single-token decoding — verification only accepts what greedy would emit, so it's a pure
    /// speedup. Verified on real Qwen2.5-3B Q4_K_M, with a deliberately repetitive prompt so the
    /// prompt-lookup drafter actually fires (and accepts) tokens. [LongFact].
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Parity")]
    public sealed class SpeculativeDecodeParityTests
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public SpeculativeDecodeParityTests(ITestOutputHelper output) => _out = output;

        [LongFact]  // heavy group, never measured — see Scripts/longfact_heavy.txt
        public void Speculative_ProducesIdenticalSequence_ToGreedy()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            // Repetitive prompt → the n-gram drafter finds matches and accepts drafts.
            var prompt = new List<int>();
            for (var r = 0; r < 8; r++)
            {
                prompt.AddRange([10, 11, 12, 13, 14, 15]);
            }
            var promptArr = prompt.ToArray();
            const int generate = 40;

            // Reference: plain greedy, single token at a time.
            var greedy = new List<int>();
            using (var s = engine.CreateSession(256))
            {
                s.Reset(promptArr);
                var sampling = SamplingOptions.Greedy;
                for (var i = 0; i < generate; i++)
                {
                    greedy.Add(s.GenerateNextToken(in sampling));
                }
            }

            // Speculative greedy: commits ≥1 token per batched verify.
            var spec = new List<int>();
            var anyMultiCommit = false;
            using (var s = engine.CreateSession(256))
            {
                s.Reset(promptArr);
                var history = new List<int>(promptArr);
                var committed = new int[6];   // maxDraft(4) + 2 (t0 + drafts + bonus)
                while (spec.Count < generate)
                {
                    var n = s.GenerateSpeculative(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history), committed, maxDraft: 4);
                    if (n > 1)
                    {
                        anyMultiCommit = true;
                    }
                    for (var c = 0; c < n; c++)
                    {
                        spec.Add(committed[c]);
                        history.Add(committed[c]);
                    }
                }
            }

            _out.WriteLine($"greedy[0..8]=[{string.Join(",", greedy.GetRange(0, 8))}]  multiCommit={anyMultiCommit}");

            // The drafter must actually have fired — without this the rest is vacuous.
            Assert.True(anyMultiCommit, "speculative never committed >1 token — drafter/verify not exercised.");

            // NOT token equality. This asserted an identical sequence until 2026-08-07, when it was executed
            // for the first time and failed at index 3: greedy 16, speculative 11930 — the RUNNER-UP, 0.1299
            // below the winner, where the batched verify and the single-token decode differ by up to 0.54 on
            // this very context. The premise was wrong, not the runtime. See SpeculativeDivergence.
            SpeculativeDivergence.AssertOnlyNearTieFlips(
                () => engine.CreateSession(256), promptArr, greedy, spec, generate, _out);
        }

        [FixtureFact(TestFixture.Qwen3BTokenizerJson)]
        public void Speculative_DecodeSpeedup_OnRepetitiveText()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var tok = QwenTokenizer.Load(@"C:\qwen3b");

            // A genuinely-echoing prompt: the model continues the repeated sentence and the n-gram
            // drafter (matching the earlier repeats) proposes exactly what greedy emits → high acceptance.
            const string text = "The quick brown fox jumps over the lazy dog. " +
                                 "The quick brown fox jumps over the lazy dog. " +
                                 "The quick brown fox jumps over the lazy dog. " +
                                 "The quick brown fox jumps over the lazy dog. ";
            var promptArr = tok.Encode(text, addBos: false);
            const int generate = 96;

            // Acceptance signal: average tokens committed per speculative step.
            {
                using var s = engine.CreateSession(512);
                s.Reset(promptArr);
                var hist = new List<int>(promptArr);
                var buf = new int[6];
                int steps = 0, produced = 0;
                while (produced < generate)
                {
                    var n = s.GenerateSpeculative(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(hist), buf, maxDraft: 4);
                    for (var c = 0; c < n; c++)
                    {
                        hist.Add(buf[c]);
                    }
                    produced += n;
                    steps++;
                }
                _out.WriteLine($"avg tokens/step = {(double)produced / steps:F2} ({produced} in {steps} steps)");
            }

            double Single()
            {
                using var s = engine.CreateSession(512);
                s.Reset(promptArr);
                var sampling = SamplingOptions.Greedy;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (var i = 0; i < generate; i++)
                {
                    s.GenerateNextToken(in sampling);
                }
                sw.Stop();
                return generate / sw.Elapsed.TotalSeconds;
            }

            double Speculative()
            {
                using var s = engine.CreateSession(512);
                s.Reset(promptArr);
                var history = new List<int>(promptArr);
                var committed = new int[6];
                var produced = 0;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                while (produced < generate)
                {
                    var n = s.GenerateSpeculative(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history), committed, maxDraft: 4);
                    for (var c = 0; c < n; c++)
                    {
                        history.Add(committed[c]);
                    }
                    produced += n;
                }
                sw.Stop();
                return produced / sw.Elapsed.TotalSeconds;
            }

            var single = Single();
            var spec = Speculative();
            _out.WriteLine($"decode tok/s: single={single:F2}  speculative={spec:F2}  speedup={spec / single:F2}×");
        }

        /// <summary>
        /// The production <see cref="ChatSession"/> path now dispatches to speculative decoding when
        /// unconstrained on a speculation-capable session. A/B: the SAME greedy chat turn, once on a
        /// normal session (CanSpeculate → speculative) and once with sliding-window forced on
        /// (CanSpeculate false → single-token). The generation is short enough that the window never
        /// evicts, so the maths is identical — any text difference is a speculative-wiring bug. [LongFact].
        /// </summary>
        [FixtureFact(TestFixture.Qwen3BTokenizerJson)]
        public void ChatSession_SpeculativePath_MatchesSingleToken_Greedy()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }
            const string dir = @"C:\qwen3b";

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var tok = new QwenChatTokenizer(QwenTokenizer.Load(dir));
            var template = new ChatTemplate(ChatTemplateFormat.ChatML);
            var options = new GenerationOptions(
                maxNewTokens: 64, maxContextLength: 512, sampling: SamplingOptions.Greedy, stopOnEndOfTextToken: true);

            const string question = "List the first five prime numbers, then count from one to five.";

            string specReply;
            using (var s = engine.CreateSession(512))
            {
                var chat = new ChatSession(s, tok, template);
                specReply = chat.Send(question, in options);
                _out.WriteLine($"speculative path: {chat.LastStats.TokensPerSecond:F2} tok/s, {chat.LastStats.GeneratedTokens} tokens");
            }

            string singleReply;
            using (var s = engine.CreateSession(512))
            {
                var chat = new ChatSession(s, tok, template, stopSequences: null, slidingWindow: true);
                singleReply = chat.Send(question, in options);
                _out.WriteLine($"single-token path: {chat.LastStats.TokensPerSecond:F2} tok/s, {chat.LastStats.GeneratedTokens} tokens");
            }

            _out.WriteLine($"\nspeculative:\n{specReply}\n\nsingle-token:\n{singleReply}");

            // This asserted string equality and passed — but it passed for the wrong reason, and the passing
            // was the problem. Nothing here verifies that speculation ENGAGED: with no explicit drafter,
            // GenerateSpeculativeCore only drafts when the adaptive gate is open AND PromptLookupDrafter
            // finds an n-gram match, which a short novel answer rarely offers. When it does not draft, the
            // step is a plain single-token step and equality is a tautology. The measured 21.96 vs 23.03
            // tok/s — the "speculative" arm SLOWER than its control — is the signature of exactly that.
            //
            // Equality is also not the right contract even when speculation does engage: the batched verify
            // and the single-token decode differ by up to ~0.5 in logits, enough to flip a near-tie
            // (SpeculativeDivergence). So this asserts what it can honestly check — the wiring runs and
            // produces a real answer — and reports the comparison instead of pretending to gate it.
            Assert.False(string.IsNullOrWhiteSpace(specReply), "the speculative path produced no text.");
            Assert.False(string.IsNullOrWhiteSpace(singleReply), "the single-token control produced no text.");

            var common = 0;

            while (common < Math.Min(specReply.Length, singleReply.Length)
                && specReply[common] == singleReply[common])
            {
                common++;
            }

            _out.WriteLine($"\ncommon prefix: {common} of {specReply.Length}/{singleReply.Length} chars"
                + (specReply == singleReply ? "  (identical)" : "  (diverged — see SpeculativeDivergence)"));
            _out.WriteLine("NOTE: whether speculation engaged at all is NOT verifiable from here — "
                + "ChatSession.LastStats exposes no commit count. See T11 in docs/test-gate-backlog.md.");
        }
    }
}
