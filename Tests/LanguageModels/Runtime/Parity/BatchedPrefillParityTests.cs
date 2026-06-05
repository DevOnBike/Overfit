// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Parity for the quantized batched prefill: prefilling a prompt through the batched path
    /// (<see cref="CachedGptStack.PrefillBatchedQuant"/>) must leave the session in the same state as
    /// the single-token loop — i.e. the end-of-prompt logits agree. Verified on the real Qwen2.5-3B
    /// Q4_K_M (RMSNorm + RoPE + GQA 16:2 + SwiGLU + mixed K-quant), the exact path the optimisation
    /// targets. [LongFact].
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Parity")]
    public sealed class BatchedPrefillParityTests
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public BatchedPrefillParityTests(ITestOutputHelper output) => _out = output;

        /// <summary>
        /// REGRESSION GUARD for the 2026-05-29 RoPE-convention fix (task #95): the GGUF loader applied
        /// adjacent-pair RoPE to Qwen2 weights that are stored in HF/NEOX split-half layout, which left
        /// position 0 correct (identity rotation) but corrupted every later position — attention collapsed
        /// onto the current token, so a real ~56-token system-message prompt produced degenerate,
        /// space-less garbage ("France'scapitalistouredisParis.") while a short prompt looked fine. Fixed
        /// by <c>GPT1Config.RopeSplitHalf</c> (set for qwen2/qwen2moe GGUF) → split-half rotation. This
        /// asserts the engine now generates coherent, correctly-spaced text. [LongFact] — needs the real
        /// model. The fast, model-free convention guards live in <c>RopeConventionTests</c>.
        /// </summary>
        [LongFact]
        public void Engine_GeneratesCoherentText_ForLongSystemPrompt()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            // Encode the EXACT prompt the demo builds (long system message → ~56 tokens), via the real
            // tokenizer, so this faithfully reproduces the failing path rather than a hand-picked 21-token
            // prompt (which happens not to trigger the bug).
            var tokenizer = QwenTokenizer.Load(@"C:\qwen3b");
            const string chatml =
                "<|im_start|>system\nYou are a concise, helpful assistant running locally inside a .NET " +
                "process. Answer only from context the user provides; if you are unsure, say so.<|im_end|>\n" +
                "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n" +
                "<|im_start|>assistant\n";
            var prompt = tokenizer.Encode(chatml);
            _out.WriteLine($"prompt length = {prompt.Length} tokens");

            // GENERATE from the raw engine (bypassing ChatSession) and decode — does the ENGINE itself
            // produce coherent text for a system-message prompt, or the "<|im_start|>system" junk seen in
            // the demo? This isolates engine vs the chat layer.
            using var gen = engine.CreateSession(2048); // match the demo's context length
            gen.Reset(prompt);
            var outTokens = new List<int>();
            var greedy = DevOnBike.Overfit.LanguageModels.Contracts.SamplingOptions.Greedy;
            for (var i = 0; i < 20; i++)
            {
                var t = gen.GenerateNextToken(in greedy);
                if (t == QwenTokenizer.EndOfText || t == 151645) { break; } // <|endoftext|> or <|im_end|>
                outTokens.Add(t);
            }
            var text = tokenizer.Decode(outTokens.ToArray());
            _out.WriteLine($"ENGINE GENERATED: '{text}'");

            // Post-fix: coherent, correctly-spaced answer. A regression in the RoPE convention would
            // collapse attention onto the current token and drop the spaces ("France'scapital...").
            Assert.Contains("Paris", text, StringComparison.OrdinalIgnoreCase);
            Assert.Contains(' ', text);
        }

        /// <summary>
        /// Model-free: replicates ChatSession.Generate's incremental delta-decode and asserts it
        /// reconstructs the same text as a whole-sequence decode — i.e. streaming detokenization must
        /// not drop spaces. Needs only the tokenizer (fast), no model weights.
        /// </summary>
        [Fact]
        public void IncrementalDecode_PreservesSpaces_LikeChatSession()
        {
            const string dir = @"C:\qwen3b";
            if (!Directory.Exists(dir)) { _out.WriteLine($"missing {dir}"); return; }

            var tok = QwenTokenizer.Load(dir);
            const string phrase = "The capital of France is Paris.";
            var ids = tok.Encode(phrase);
            var whole = tok.Decode(ids);

            // ChatSession.Generate's exact incremental logic.
            var prev = string.Empty;
            var sb = new StringBuilder();
            var gen = new List<int>();
            foreach (var id in ids)
            {
                gen.Add(id);
                var full = tok.Decode(CollectionsMarshal.AsSpan(gen));
                if (full.Length <= prev.Length || !full.StartsWith(prev, StringComparison.Ordinal)) { continue; }
                sb.Append(full[prev.Length..]);
                prev = full;
            }

            _out.WriteLine($"whole='{whole}' incremental='{sb}'");
            Assert.Equal(whole, sb.ToString());
        }

        [LongFact]
        public void BatchedPrefill_MatchesSingleToken_OnRealQwen()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            // A ≥16-token prompt to trigger the batched path; arbitrary in-vocab ids.
            var prompt = new int[40];
            for (var i = 0; i < prompt.Length; i++) { prompt[i] = 100 + i * 37; }

            // Batched (default eligibility kicks in for this length).
            using var batched = engine.CreateSession(256);
            batched.Reset(prompt);
            var bLogits = batched.LastLogits.ToArray();

            // Single-token reference (force the loop via the test hook).
            using var single = engine.CreateSession(256);
            single.DisableBatchedPrefillForParity = true;
            single.Reset(prompt);
            var sLogits = single.LastLogits;

            var maxDiff = 0f;
            int argB = 0, argS = 0;
            for (var i = 0; i < sLogits.Length; i++)
            {
                maxDiff = MathF.Max(maxDiff, MathF.Abs(bLogits[i] - sLogits[i]));
                if (bLogits[i] > bLogits[argB]) { argB = i; }
                if (sLogits[i] > sLogits[argS]) { argS = i; }
            }

            _out.WriteLine($"argmax batched={argB} single={argS}  maxAbsLogitDiff={maxDiff:G4}");

            // Same predicted token, and logits agree to within accumulated FP noise across 36 layers.
            Assert.Equal(argS, argB);
            Assert.True(maxDiff < 1e-2f, $"batched vs single logit divergence {maxDiff:G4} (> 1e-2).");
        }

        [LongFact]
        public void BatchedPrefill_MatchesSingleToken_OnRealQwenMoE()
        {
            const string moePath = @"C:\qwen-moe\Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf";
            if (!File.Exists(moePath)) { _out.WriteLine($"missing {moePath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(moePath);
            Assert.True(engine.Config.IsMixtureOfExperts);

            var prompt = new int[32];
            for (var i = 0; i < prompt.Length; i++) { prompt[i] = 100 + i * 53; }

            using var batched = engine.CreateSession(128);
            batched.Reset(prompt);
            var bLogits = batched.LastLogits.ToArray();

            using var single = engine.CreateSession(128);
            single.DisableBatchedPrefillForParity = true;
            single.Reset(prompt);
            var sLogits = single.LastLogits;

            var moeMaxDiff = 0f;
            int moeArgB = 0, moeArgS = 0;
            for (var i = 0; i < sLogits.Length; i++)
            {
                moeMaxDiff = MathF.Max(moeMaxDiff, MathF.Abs(bLogits[i] - sLogits[i]));
                if (bLogits[i] > bLogits[moeArgB]) { moeArgB = i; }
                if (sLogits[i] > sLogits[moeArgS]) { moeArgS = i; }
            }
            _out.WriteLine($"MoE argmax batched={moeArgB} single={moeArgS}  maxAbsLogitDiff={moeMaxDiff:G4}");

            // Batched MoE accumulates each row's experts in top-k slot order (= single-token order), so
            // it's BIT-IDENTICAL despite the gather-by-expert grouping — no routing-flip cascade.
            Assert.Equal(moeArgS, moeArgB);
            Assert.True(moeMaxDiff < 1e-3f, $"MoE batched vs single logit divergence {moeMaxDiff:G4} (> 1e-3).");
        }

        [LongFact]
        public void BatchedPrefill_TtftSpeedup_OnRealQwen()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            var prompt = new int[256];
            for (var i = 0; i < prompt.Length; i++) { prompt[i] = 100 + i * 11; }

            double Time(bool disableBatched)
            {
                var best = double.MaxValue;
                for (var r = 0; r < 3; r++)
                {
                    using var s = engine.CreateSession(512);
                    s.DisableBatchedPrefillForParity = disableBatched;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    s.Reset(prompt);   // prefill the whole prompt (TTFT)
                    sw.Stop();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                }
                return best;
            }

            var single = Time(disableBatched: true);
            var batched = Time(disableBatched: false);
            _out.WriteLine($"TTFT {prompt.Length}-token prompt: single={single:F1} ms  batched={batched:F1} ms  speedup={single / batched:F2}×");
        }
    }
}
