// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

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
        [LongFact("2s")]
        public void Engine_GeneratesCoherentText_ForLongSystemPrompt()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }

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
                if (t == QwenTokenizer.EndOfText || t == 151645)
                {
                    break;
                } // <|endoftext|> or <|im_end|>
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
            // A skip, not a return: without the tokenizer this test asserts nothing, and a pass would be
            // indistinguishable from "checked and correct" on a box with no fixtures (i.e. on CI).
            // Gated on the FILE, not on the directory: an existing but empty fixture directory passes a
            // Directory.Exists check and then throws inside QwenTokenizer.Load — measured, that arm went
            // red rather than skipping.
            var dir = TestModelPaths.Qwen3B.Dir;
            Assert.SkipWhen(!File.Exists(TestModelPaths.Qwen3B.TokenizerJsonPath),
                $"tokenizer.json not present in {dir} (set OVERFIT_QWEN3B_DIR).");

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
                if (full.Length <= prev.Length || !full.StartsWith(prev, StringComparison.Ordinal))
                {
                    continue;
                }
                sb.Append(full[prev.Length..]);
                prev = full;
            }

            _out.WriteLine($"whole='{whole}' incremental='{sb}'");
            Assert.Equal(whole, sb.ToString());
        }

        [LongFact("4s")]
        public void BatchedPrefill_MatchesSingleToken_OnRealQwen()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }

            // The scope is entered BEFORE the load, and the order is load-bearing since 2026-08-21: the
            // loader reads this flag to decide whether to build the per-head attention output weights at all
            // (GgufLlamaLoader.UseWholeOutputOnly), and an array it skipped cannot be produced afterwards.
            // Loading first and scoping second throws a named error rather than silently comparing arms.
            using var kernels = UseNonRepackedKernels();
            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            // A ≥16-token prompt to trigger the batched path; arbitrary in-vocab ids.
            var prompt = new int[40];
            for (var i = 0; i < prompt.Length; i++)
            {
                prompt[i] = 100 + i * 37;
            }

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
                if (bLogits[i] > bLogits[argB])
                {
                    argB = i;
                }
                if (sLogits[i] > sLogits[argS])
                {
                    argS = i;
                }
            }

            _out.WriteLine($"argmax batched={argB} single={argS}  maxAbsLogitDiff={maxDiff:G4}");

            // Same predicted token, and logits agree to within accumulated FP noise across 36 layers.
            Assert.Equal(argS, argB);
            Assert.True(maxDiff < 1e-2f, $"batched vs single logit divergence {maxDiff:G4} (> 1e-2).");
        }

        /// <summary>
        /// Guards the DEFAULT prefill configuration — the repacked <c>block_q*_Kx8</c> GEMMs — at the standard
        /// they can actually meet: <b>same predicted token</b>, not bit-equality.
        ///
        /// <para>Those kernels associate their reduction differently from the per-row ones, so they diverge
        /// from the single-token reference by ~0.44 in absolute logits on Qwen-3B. That is why
        /// <see cref="BatchedPrefill_MatchesSingleToken_OnRealQwen"/> pins the layout via
        /// <see cref="UseNonRepackedKernels"/> — and why the fast path needs its own, looser gate rather than
        /// simply being untested. Without this test, turning a repacked kernel on by default would be covered
        /// by nothing at all.</para>
        ///
        /// <para>The tolerance is deliberately not tightened to the point of pinning today's exact numerics:
        /// the contract being asserted is "the reassociation does not change what the model says", which is
        /// the same bar <c>OVERFIT_REPACK_ATTN</c> is held to.</para>
        ///
        /// <para><b>The prompt is real text, not the synthetic id ramp its neighbours use.</b> An arbitrary
        /// in-vocab sequence like <c>100 + 37·i</c> is out-of-distribution, so the top logits come out nearly
        /// tied and the argmax flips on any numerical perturbation — this test failed exactly that way with
        /// the ramp (argmax 11 vs 13 at maxAbsLogitDiff 0.42) while the same kernels agree on the first
        /// generated token for real text. Argmax stability is only a meaningful assertion where the model is
        /// actually confident.</para>
        /// </summary>
        [LongFact("2s")]
        public void RepackedPrefill_AgreesWithNonRepacked_OnArgmax()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
            var tokenizer = GgufTokenizer.Load(ModelPath);
            var prompt = tokenizer.Encode(
                "The history of computing began with mechanical calculators and evolved through vacuum tubes, "
                + "transistors, integrated circuits and finally the microprocessor era. The next paragraph "
                + "explains why that progression mattered for modern software.");

            // Default configuration: whatever the repacked gates decide (sidecar / env flag / Q6_K tiled).
            using var fast = engine.CreateSession(256);
            fast.Reset(prompt);
            var fastLogits = fast.LastLogits.ToArray();

            // The reference arm needs its OWN engine, loaded inside the scope. Since 2026-08-21 the loader
            // reads this flag to decide whether the per-head attention output weights are built at all
            // (GgufLlamaLoader.UseWholeOutputOnly), so the engine above — loaded outside the scope — does not
            // carry them and cannot serve as the non-repacked reference. Two loads of the same file give
            // identical weights, so the comparison is unchanged; it costs one extra load in a [LongFact].
            float[] referenceLogits;
            using (var kernels = UseNonRepackedKernels())
            {
                using var referenceEngine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);
                using var reference = referenceEngine.CreateSession(256);
                reference.Reset(prompt);
                referenceLogits = reference.LastLogits.ToArray();
            }

            var maxDiff = 0f;
            int argFast = 0, argReference = 0;
            for (var i = 0; i < referenceLogits.Length; i++)
            {
                maxDiff = MathF.Max(maxDiff, MathF.Abs(fastLogits[i] - referenceLogits[i]));
                if (fastLogits[i] > fastLogits[argFast])
                {
                    argFast = i;
                }
                if (referenceLogits[i] > referenceLogits[argReference])
                {
                    argReference = i;
                }
            }

            _out.WriteLine(
                $"repacked argmax={argFast} non-repacked argmax={argReference}  maxAbsLogitDiff={maxDiff:G4}");

            Assert.Equal(argReference, argFast);
        }

        [LongFact("12s")]
        public void BatchedPrefill_MatchesSingleToken_OnRealQwenMoE()
        {
            const string moePath = @"C:\qwen-moe\Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf";
            if (!File.Exists(moePath))
            {
                _out.WriteLine($"missing {moePath}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(moePath);
            Assert.True(engine.Config.IsMixtureOfExperts);

            var prompt = new int[32];
            for (var i = 0; i < prompt.Length; i++)
            {
                prompt[i] = 100 + i * 53;
            }

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
                if (bLogits[i] > bLogits[moeArgB])
                {
                    moeArgB = i;
                }
                if (sLogits[i] > sLogits[moeArgS])
                {
                    moeArgS = i;
                }
            }
            _out.WriteLine($"MoE argmax batched={moeArgB} single={moeArgS}  maxAbsLogitDiff={moeMaxDiff:G4}");

            // Batched MoE accumulates each row's experts in top-k slot order (= single-token order), so
            // it's BIT-IDENTICAL despite the gather-by-expert grouping — no routing-flip cascade.
            Assert.Equal(moeArgS, moeArgB);
            Assert.True(moeMaxDiff < 1e-3f, $"MoE batched vs single logit divergence {moeMaxDiff:G4} (> 1e-3).");
        }

        [LongFact("39s")]
        public void BatchedPrefill_TtftSpeedup_OnRealQwen()
        {
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"missing {ModelPath}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            var prompt = new int[256];
            for (var i = 0; i < prompt.Length; i++)
            {
                prompt[i] = 100 + i * 11;
            }

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
        /// <summary>
        /// Forces the NON-repacked batched kernels for the duration of the scope — see
        /// <see cref="NonRepackedKernelScope"/>, which carries the reasoning and is now the only writer of
        /// the flag in the test tree. This wrapper is kept only so the call sites below still read as prose.
        ///
        /// <para>The private near-twin that used to live here reset the flag to <c>false</c> rather than
        /// restoring it, and the flag was process-global: with collections running in parallel it switched
        /// the kernel under other tests mid-assertion.</para>
        /// </summary>
        private static NonRepackedKernelScope UseNonRepackedKernels() => new();
    }
}
