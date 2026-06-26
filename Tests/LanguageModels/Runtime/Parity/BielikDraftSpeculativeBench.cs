// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Draft-MODEL speculative decoding for Bielik on the pure-CPU stack — DOSSIER of a measured NET LOSS, on the
    /// very models the SpeakLeash DFlash release targets. Bielik-1.5B-v3.0 (draft) speculates for Bielik-4.5B-v3.0
    /// (target), SAME v3.0 tokenizer/vocab. Motivated by DFlash (dedicated Bielik draft models for vLLM/SGLang
    /// claiming "kilkukrotnie szybciej") — this tests whether the technique transfers to our CPU stack.
    ///
    /// MEASURED 2026-06-23 (16-core dev box, Q4_K_M, 80 tok, greedy, novel Polish prompt):
    ///   single-token 4.5B decode: 16.45 tok/s (baseline)
    ///   maxDraft=2: 3.42 tok/step accepted, 11.13 tok/s → 0.68×
    ///   maxDraft=4: 3.86 tok/step,           9.53 tok/s → 0.58×
    ///   maxDraft=6: 4.61 tok/step,           9.19 tok/s → 0.56×
    ///   maxDraft=8: 4.72 tok/step,           7.96 tok/s → 0.48×
    ///
    /// FINDING: a NET LOSS (0.48–0.68×) despite GOOD acceptance (3.4–4.7 tok/step). Same structural reason as
    /// <see cref="DraftModelSpeculativeBench"/> (Qwen 0.5B→3B): our batched verify is COMPUTE-bound, so verifying K
    /// draft tokens costs ≈ K× a single token → the speedup ceiling is low even with free acceptance, and running a
    /// real 1.5B draft maxDraft times/step eats it. Draft-model speculation only pays off when the verify is
    /// BANDWIDTH-bound — GPU (vLLM/SGLang, where DFlash wins), or a much larger target (Bielik-11B, the actual DFlash
    /// target, not tested here). On CPU the free-draft prompt-lookup path (PromptLookupDrafter) is the one that wins.
    /// The real CPU lever would be making the K-token verify amortize weight reads (a prefill/verify GEMM project),
    /// NOT adding a draft model. Asserts greedy bit-identity. [LongFact] — flip to [Fact] to re-measure.
    /// </summary>
    [Trait("Category", "Bielik")]
    [Trait("Category", "Parity")]
    public sealed class BielikDraftSpeculativeBench
    {
        private const string TargetGguf = @"C:\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf";
        private const string DraftGguf = @"C:\bielik\bielik-1.5b-v3.0-instruct-q4_k_m-imat.gguf";

        private readonly ITestOutputHelper _out;
        public BielikDraftSpeculativeBench(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Bielik_DraftModel_Speculative_BitIdentical_AndSpeedup()
        {
            if (!File.Exists(TargetGguf) || !File.Exists(DraftGguf))
            {
                _out.WriteLine("missing target or draft gguf");
                return;
            }

            using var target = CachedLlamaInferenceEngine.LoadGguf(TargetGguf);
            using var draft = CachedLlamaInferenceEngine.LoadGguf(DraftGguf);
            var tok = GgufTokenizer.Load(TargetGguf);

            const string text =
                "Wyjaśnij w trzech zdaniach, dlaczego niebo jest niebieskie w ciągu dnia, a czerwone o zachodzie słońca.";
            var promptArr = tok.Encode(text, addBos: true);
            const int generate = 80;

            // ── Reference: plain greedy single-token on the target ──
            var refSeq = new List<int>(generate);
            double singleTokPerSec;
            using (var s = target.CreateSession(1024))
            {
                s.Reset(promptArr);
                var g = SamplingOptions.Greedy;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (var i = 0; i < generate; i++)
                {
                    refSeq.Add(s.GenerateNextToken(in g));
                }
                sw.Stop();
                singleTokPerSec = generate / sw.Elapsed.TotalSeconds;
            }
            _out.WriteLine($"single-token target (4.5B) decode: {singleTokPerSec:F2} tok/s");

            // ── Draft-model speculative (greedy), swept over maxDraft ──
            foreach (var maxDraft in new[] { 2, 4, 6, 8 })
            {
                var specSeq = new List<int>(generate + maxDraft);
                var steps = 0;
                double specTokPerSec;
                using (var s = target.CreateSession(1024))
                using (var d = draft.CreateSession(1024))
                {
                    s.Reset(promptArr);
                    var drafter = new DraftModelSpeculativeDrafter(d, promptArr);
                    var history = new List<int>(promptArr);
                    var committed = new int[maxDraft + 2];
                    var g = SamplingOptions.Greedy;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    while (specSeq.Count < generate)
                    {
                        var n = s.GenerateSpeculative(CollectionsMarshal.AsSpan(history), committed, in g, maxDraft, drafter);
                        for (var c = 0; c < n; c++)
                        {
                            specSeq.Add(committed[c]);
                            history.Add(committed[c]);
                        }
                        steps++;
                    }
                    sw.Stop();
                    specTokPerSec = specSeq.Count / sw.Elapsed.TotalSeconds;
                }

                _out.WriteLine($"maxDraft={maxDraft}: avg {(double)specSeq.Count / steps:F2} tok/step, " +
                               $"draft-spec {specTokPerSec:F2} tok/s, speedup {specTokPerSec / singleTokPerSec:F2}×");

                for (var i = 0; i < generate; i++)
                {
                    Assert.Equal(refSeq[i], specSeq[i]);
                }
            }
        }
    }
}
