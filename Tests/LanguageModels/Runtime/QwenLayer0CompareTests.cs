// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenL0")]
    [Trait("Category", "Qwen")]
    public sealed class QwenLayer0CompareTests
    {
        private readonly ITestOutputHelper _out;
        public QwenLayer0CompareTests(ITestOutputHelper output) => _out = output;
        private static string ModelPath => TestModelPaths.Qwen3B.BinaryPath;
        // Original code had TokenizerDir = "c:/qwen/" (typo — pointed at a non-existent
        // sibling dir). The model + tokenizer live under the same Qwen3B root.
        private static string TokenizerDir => TestModelPaths.Qwen3B.Dir;

        /// <summary>
        /// Position-0 (BOS): C# logits must match Python forward_multitoken.py.
        /// Current fixture: Qwen2.5-3B-Instruct FP16 (36 layers, head_dim=128).
        /// Python TEST 1 → top-1 = [33975] 15.5608.
        /// </summary>
        [LongFact("5s")]
        public void L0_LogitsAfterReset_NotAfterGenerate()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                session.Reset([151643]);
                var logits = session.LastLogits.ToArray();

                _out.WriteLine("=== LOGITS AFTER RESET (position 0) ===");
                var top5 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(5).ToArray();
                _out.WriteLine("C# TOP-5:");
                foreach (var (v, i) in top5)
                {
                    _out.WriteLine($"  [{i,7}]  {v,8:F4}");
                }

                // Top-1 expected: [33975] 15.5608 (from forward_multitoken.py TEST 1 for 3B FP16)
                var top1 = top5[0];
                _out.WriteLine($"C# top-1   = [{top1.i}] {top1.v:F4}");
                _out.WriteLine("Python top-1 = [33975] 15.5608");

                Assert.Equal(33975, top1.i);
                Assert.True(Math.Abs(top1.v - 15.5608f) < 0.1f,
                    $"logit[33975]={top1.v:F4} should be ≈15.5608 (got diff={top1.v - 15.5608f:F4})");
            }
        }

        /// <summary>
        /// C# hidden state and logits vs the Python oracle for 2 tokens <c>[BOS, im_start]</c>.
        ///
        /// <para><b>PROVENANCE — read this before touching the numbers.</b> The reference lives in
        /// <c>Tests/test_fixtures/qwen3b_l0_twotoken_hidden.json</c> and was recorded <b>2026-08-15</b> by
        /// <c>Scripts/forward_multitoken.py</c>'s <c>forward_sequence</c> (adjacent-pair / NEOX RoPE, grouped
        /// GQA) against <c>C:\qwen3b\qwen.bin</c>, mtime <b>2026-08-07T12:29:37Z</b>, <b>13 593 731 124</b> bytes.
        /// The fixture repeats all of that in its own header, and <see cref="LoadTwoTokenOracle"/>
        /// <i>asserts</i> both against the file on disk — so a re-converted model fails by name instead of as
        /// an unexplained cosine miss. Re-record with <c>Scripts/oracle_twotoken.py</c>.</para>
        ///
        /// <para><b>Why it was re-recorded (XC-55).</b> The previous constants — top-1 <c>[198] 12.3511</c>,
        /// <c>hidden[:4] = [0.14059, 0.84549, 1.01591, -1.83366]</c> — were taken 2026-05-17 and describe the
        /// file this one <i>replaced</i>: the .bin was re-converted on 2026-08-07, after commit
        /// <c>265fd77</c> added <c>permute_rope_rows</c> to <c>Scripts/convert_llama.py</c> (the HF
        /// rotate-half → adjacent-pair permute of the Q/K weights and biases). <b>Position 0 is the identity
        /// rotation</b>, which is why the 1-token oracle in
        /// <see cref="L0_LogitsAfterReset_NotAfterGenerate"/> survived the fixture change untouched and only
        /// position ≥ 1 diverged. The re-recording run reproduced that 1-token oracle exactly
        /// (<c>[33975] 15.5608</c>) as its own self-check, so the parse behind these numbers is verified.</para>
        ///
        /// <para><b>The accessor was NOT the defect</b>, and this test is no longer the only thing saying so:
        /// <see cref="LastHiddenState_ReproducesLastLogits_SingleTokenPrefill"/> pins "the exposed tensor is
        /// the one the logits came from" <i>without</i> an oracle, so it survives the next re-conversion.
        /// This test is the perishable half.</para>
        ///
        /// <para><b>Cosine, not absolute distance, on the hidden state.</b> The two sides differ in weight
        /// precision and reduction order, and this prompt is two special tokens — out of distribution, and
        /// numerically ill-conditioned in exactly the way
        /// <c>BatchedPrefillParityTests.RepackedPrefill_AgreesWithNonRepacked_OnArgmax</c> already records for
        /// argmax. An absolute bound over four components would break again on any precision-touching change
        /// and the next reader could not tell that from a real defect. <b>Measured 2026-08-15 on this box:
        /// cosine over all 2048 components = 0.99999992</b>, against the bound of 0.9999 below; the largest
        /// absolute component gap in <c>hidden[:4]</c> was 0.00178, and <c>logit[198]</c> read 11.3741 against
        /// the oracle's 11.3747. The four-component absolute check is kept only as a sanity floor.</para>
        /// </summary>
        [LongFact("7s")]
        public void L0_TwoToken_HiddenStateVsPython()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();

            var reference = LoadTwoTokenOracle();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                session.Reset([151643, 151644]);

                var logits = session.LastLogits.ToArray();
                var top1 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();

                _out.WriteLine("=== 2-TOKEN [BOS, im_start] ===");
                _out.WriteLine($"C# top-1 = [{top1.i}] {top1.v:F4}");
                _out.WriteLine($"  Python:  [198] {reference.Logit198:F4}  (grouped GQA, oracle re-recorded 2026-08-15)");
                _out.WriteLine($"  Match:   {(top1.i == 198 ? "SAME TOKEN" : $"DIFFERENT (got {top1.i})")}");
                _out.WriteLine(string.Empty);

                // HIDDEN STATE comparison
                var hidden = session.LastHiddenState.ToArray();
                Assert.Equal(reference.Hidden.Length, hidden.Length);

                var cosine = Cosine(hidden, reference.Hidden);
                _out.WriteLine("=== HIDDEN STATE (before final RMSNorm) ===");
                _out.WriteLine($"C# hidden[:4] = [{string.Join(", ", hidden.Take(4).Select(v => v.ToString("F5")))}]");
                _out.WriteLine($"Py hidden[:4] = [{string.Join(", ", reference.Hidden.Take(4).Select(v => v.ToString("F5")))}]");
                _out.WriteLine($"cosine over all {hidden.Length} components = {cosine:F8}");

                var maxDiff = 0f;
                for (var i = 0; i < 4; i++)
                {
                    var diff = hidden[i] - reference.Hidden[i];
                    _out.WriteLine($"  [{i}]: C#={hidden[i]:F5}  Py={reference.Hidden[i]:F5}  diff={diff:+0.00000;-0.00000}");
                    maxDiff = Math.Max(maxDiff, Math.Abs(diff));
                }
                _out.WriteLine($"Max |diff| hidden[:4] = {maxDiff:F5}");

                Assert.Equal(198, top1.i);

                // The top-1 VALUE, which nothing asserted before XC-55 — it read 11.3741 against a docstring
                // claiming 12.3511, ~8% apart, unchecked. Tolerance chosen the way L0_LogitsAfterReset chose
                // its own 0.1.
                Assert.True(Math.Abs(top1.v - reference.Logit198) < 0.1f,
                    $"logit[198]={top1.v:F4} should be ~{reference.Logit198:F4} (diff={top1.v - reference.Logit198:F4})");

                Assert.True(cosine > 0.9999,
                    $"Hidden state must match the Python oracle in direction; cosine={cosine:F8}");
                Assert.True(maxDiff < 0.05f,
                    $"Sanity floor on hidden[:4] absolute agreement, got {maxDiff:F5}");
            }
        }

        /// <summary>
        /// ORACLE-FREE self-consistency, at a prompt short enough to take the single-token prefill loop:
        /// projecting <c>session.LastHiddenState</c> through <c>engine.LogitLens</c> must reproduce
        /// <c>session.LastLogits</c>.
        ///
        /// <para>This is what nobody had when <c>XC-55</c> was opened. The task's hypothesis — that the
        /// accessor exposes a different tensor from the one the logits were computed from (a wrong row, or a
        /// buffer overwritten after the last layer) — was argued about for a day against constants that turned
        /// out to be stale. This assertion answers it directly and <b>needs no Python</b>, so it survives
        /// every future re-conversion of the fixture. The lens applies the same <c>ApplyFinalNorm</c> +
        /// <c>ProjectLogitsFrom</c> the real next-token projection uses, so agreement is near-tautological
        /// against the code as it reads today — which is the point: it is a change detector for exactly the
        /// edit that would break it. <b>Measured 2026-08-15, both prompt lengths: <c>max|logit - lens|</c> is
        /// exactly 0</b> — <c>ApplyFinalNorm</c> and <c>CachedGptStack.FinalNorm</c> are the same arithmetic
        /// and the F32 head projection is deterministic. The 1e-3 bound is headroom for a future
        /// reassociation, not a measured spread.</para>
        ///
        /// <para><b>One session per engine, deliberately.</b> <c>CachedLlamaInferenceEngine.CreateSession</c>
        /// hands every session the engine's single <c>CachedGptStack</c>, so a second live session would
        /// overwrite the hidden state this test reads. That sharing is a real defect on a public API and is
        /// filed separately; here it is simply avoided.</para>
        /// </summary>
        [LongFact("7s")]
        public void LastHiddenState_ReproducesLastLogits_SingleTokenPrefill()
        {
            // 2 tokens < CachedLlamaSession.BatchedPrefillThreshold (16), so this takes the single-token
            // loop: CachedGptStack.DecodeWithoutLogits, which writes _lastFinalHidden and _finalHidden from
            // the same `current`.
            AssertLogitLensReproducesLogits([151643, 151644], "single-token prefill loop");
        }

        /// <summary>
        /// The same oracle-free self-consistency at a prompt long enough to take the BATCHED prefill —
        /// <c>CachedGptStack.PrefillBatchedQuant</c>, whose <c>rows - 1</c> is the row index the task's own
        /// hypothesis named and which no test reached. See
        /// <see cref="LastHiddenState_ReproducesLastLogits_SingleTokenPrefill"/> for what this pins and why
        /// it is worth more than the oracle it sits beside.
        ///
        /// <para><b>What it does NOT cover, measured rather than assumed.</b> It is a <i>self</i>-consistency
        /// check, so it is blind to any change that moves the exposed hidden and the logits together. Mutating
        /// <c>rows - 1</c> to <c>0</c> in <c>PrefillBatchedQuant</c> does exactly that — the one <c>lastRow</c>
        /// local feeds both <c>_lastFinalHidden</c> and the final norm — and this test stayed <b>green</b>
        /// (2026-08-15, <c>max|logit - lens|</c> still 0, argmax merely shifted 17 → 369). What catches that
        /// mutation is <c>BatchedPrefillParityTests.BatchedPrefill_MatchesSingleToken_OnRealQwen</c>
        /// (<c>maxAbsLogitDiff</c> 16.31). Desynchronising the two — exposing row 0 while the logits keep the
        /// last row, which is the defect this test exists for — reddens it alone, at
        /// <c>max|logit - lens| = 31.7</c>.</para>
        /// </summary>
        [LongFact("8s")]
        public void LastHiddenState_ReproducesLastLogits_BatchedPrefill()
        {
            // 26 real chat tokens ("What is 2+2?" with a system message) — the same sequence
            // Scripts/forward_multitoken.py drives as its TEST 3. >= 16, so Prefill takes the batched branch.
            AssertLogitLensReproducesLogits(
                [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198,
                 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198],
                "batched prefill (PrefillBatchedQuant)");
        }

        private void AssertLogitLensReproducesLogits(int[] prompt, string path)
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();

            using var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using var session = engine.CreateSession(64);
            session.Reset(prompt);

            var logits = session.LastLogits.ToArray();
            var lens = new float[logits.Length];
            engine.LogitLens(session.LastHiddenState, lens);

            var maxDiff = 0f;
            int argLogits = 0, argLens = 0;
            for (var i = 0; i < logits.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(logits[i] - lens[i]));
                if (logits[i] > logits[argLogits])
                {
                    argLogits = i;
                }
                if (lens[i] > lens[argLens])
                {
                    argLens = i;
                }
            }

            _out.WriteLine($"{prompt.Length} tokens via {path}: argmax logits={argLogits} lens={argLens}  " +
                           $"max|logit - lens| = {maxDiff:G6}");

            Assert.Equal(argLogits, argLens);
            Assert.True(maxDiff < 1e-3f,
                $"LogitLens(LastHiddenState) must reproduce LastLogits — the exposed hidden IS the one the " +
                $"logits came from. Got max|diff| {maxDiff:G6} via {path}.");
        }

        /// <summary>
        /// The re-recorded Python reference: the whole pre-final-norm hidden vector plus <c>logit[198]</c>.
        /// Committed under <c>Tests/test_fixtures/</c> (copied to the build output) rather than written as
        /// 2048 literals, and it carries its own provenance header — see
        /// <see cref="L0_TwoToken_HiddenStateVsPython"/>.
        ///
        /// <para><b>It asserts the header's model identity before returning the numbers.</b> An oracle is only
        /// valid for the exact <c>.bin</c> it was recorded against, and that is the defect XC-55 existed to
        /// close: the previous constants were taken 2026-05-17 and silently described the file the 2026-08-07
        /// re-conversion replaced, so the rot presented as an unexplained cosine miss and cost a day. With the
        /// check it presents as a named failure that says what to run.</para>
        /// </summary>
        private static (float[] Hidden, float Logit198) LoadTwoTokenOracle()
        {
            var path = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "qwen3b_l0_twotoken_hidden.json");
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Missing 2-token oracle fixture '{path}'. It is committed under Tests/test_fixtures/ and " +
                    "copied to the test build output; regenerate with Scripts/oracle_twotoken.py.", path);
            }

            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            var root = doc.RootElement;

            var model = new FileInfo(ModelPath);
            var recordedBytes = root.GetProperty("model_bytes").GetInt64();
            var recordedMtime = root.GetProperty("model_mtime_utc").GetDateTimeOffset().UtcDateTime;
            // Tolerance, not equality, on the timestamp only: Python records `os.stat().st_mtime` — a float64,
            // truncated to microseconds by isoformat() — while .NET reads 100 ns ticks, so the same untouched
            // file differs in the last digits (.968373Z recorded vs .9683732Z on disk). The byte length is the
            // exact half of the pair; a re-conversion moves the mtime by hours, not by a microsecond.
            var skew = (model.LastWriteTimeUtc - recordedMtime).Duration();

            Assert.True(model.Length == recordedBytes && skew < TimeSpan.FromSeconds(1),
                $"The oracle in '{path}' was recorded against a different model file, so its numbers do not " +
                $"describe '{ModelPath}'. Re-record it: run `python Scripts/oracle_twotoken.py`, check its " +
                "SELF-CHECK line prints PASS, then copy Tests/bin/xc55-oracle.json over " +
                "Tests/test_fixtures/qwen3b_l0_twotoken_hidden.json. " +
                $"Recorded: {recordedBytes} bytes, mtime {recordedMtime:O}. " +
                $"On disk: {model.Length} bytes, mtime {model.LastWriteTimeUtc:O}.");

            var hidden = root.GetProperty("hidden").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            return (hidden, root.GetProperty("logit_198").GetSingle());
        }

        private static double Cosine(float[] a, float[] b)
        {
            double dot = 0, na = 0, nb = 0;
            for (var i = 0; i < a.Length; i++)
            {
                dot += (double)a[i] * b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
            }
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        }

        /// <summary>
        /// Full chat prompt (36 tokens with a correct system message).
        /// C# and Python agree: top-1 = [36366] ≈ 11.9
        /// </summary>
        [LongFact("10s")]
        public void L0_ChatPromptLogits()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var tok = QwenTokenizer.Load(TokenizerDir);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                var fullPrompt = tok.BuildChatPrompt("What is 2+2?");
                session.Reset(fullPrompt);
                var logits = session.LastLogits.ToArray();

                _out.WriteLine($"=== CHAT PROMPT LOGITS (position {fullPrompt.Length - 1}) ===");
                _out.WriteLine($"Prompt: {fullPrompt.Length} tokens");
                _out.WriteLine("C# TOP-10:");
                var top10 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(10).ToArray();
                for (var r = 0; r < top10.Length; r++)
                {
                    var (v, id) = top10[r];
                    var dec = tok.DecodeToken(id).Replace("\n", "\\n");
                    _out.WriteLine($"  #{r + 1,2}  [{id,7}]  {v,8:F3}  '{dec}'");
                }
                _out.WriteLine($"logit[19]=' 4'  = {logits[19],8:F4}");
                _out.WriteLine($"logit[220]=' ' = {logits[220],8:F4}");
                _out.WriteLine(string.Empty);
                _out.WriteLine("Python top-1: [151644] 22.36 (post-fix 3B FP16 grouped GQA)");
            }
        }

        /// <summary>
        /// Progressive prefix: when does the token '4' appear as top-1.
        /// </summary>
        [LongFact("1min33s")]
        public void Multitoken_ProgressivePrefixTest()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();
            TestModelPaths.Qwen3B.RequireTokenizerJsonPath();

            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            var tok = QwenTokenizer.Load(TokenizerDir);
            using (engine)
            {
                using var session = engine.CreateSession(64);
                var fullPrompt = tok.BuildChatPrompt("What is 2+2?");
                _out.WriteLine($"Full chat prompt: {fullPrompt.Length} tokens");
                _out.WriteLine($"Tokens: [{string.Join(", ", fullPrompt)}]");
                _out.WriteLine(string.Empty);

                _out.WriteLine($"{"Len",4}  {"Top-1 Token",8}  {"Top-1 Logit",10}  {"Top-1 Decoded",-20}");
                _out.WriteLine(new string('-', 55));

                for (var n = 1; n <= Math.Min(fullPrompt.Length, 36); n++)
                {
                    session.Reset(fullPrompt.Take(n).ToArray());
                    var logits = session.LastLogits.ToArray();
                    var top1 = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).First();
                    var dec = tok.DecodeToken(top1.i).Replace("\n", "\\n");
                    _out.WriteLine($"{n,4}  [{top1.i,7}]  {top1.v,10:F3}  '{dec}'");
                }
            }
        }
    }
}
