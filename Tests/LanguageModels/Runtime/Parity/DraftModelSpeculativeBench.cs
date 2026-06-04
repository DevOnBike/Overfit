// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Draft-MODEL speculative decoding — DOSSIER of a measured NET LOSS on the pure-CPU stack. A small
    /// Qwen2.5-0.5B (safetensors, SAME vocabulary) drafts for the Qwen2.5-3B Q4_K_M target. The draft is the
    /// co-located <c>C:\qwen3b\model.safetensors</c> (config.json = hidden 896 / 24 layers = the 0.5B), which
    /// shares the 3B's tokenizer.
    ///
    /// FINDINGS (this test asserts the first, measures+prints the second):
    /// (a) greedy output is BIT-IDENTICAL to single-token decode — the verify only accepts what the target
    ///     would emit, for every maxDraft (the asserted correctness invariant).
    /// (b) it is SLOWER, not faster: measured speedup maxDraft 2→0.66×, 4→0.64×, 6→0.55× despite good
    ///     acceptance (3–4 tok/step). STRUCTURAL: the batched verify is compute-bound (~4.4× a single token
    ///     at batch≈6 → ~1.36× ceiling even with a FREE draft); a real draft model adds ~6 small-model
    ///     forwards/step, which exceeds that headroom. Prompt-lookup wins precisely because its draft is free.
    ///     Draft-model speculative only pays off when the verify is BANDWIDTH-bound (GPU / much larger target).
    ///
    /// Kept as a dossier so the negative result is reproducible and not re-litigated. [LongFact].
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Parity")]
    public sealed class DraftModelSpeculativeBench
    {
        private const string TargetGguf = @"C:\qwen3b\qwen.q4km.gguf";
        private const string DraftDir = @"C:\qwen3b";   // config.json + model.safetensors = Qwen2.5-0.5B

        private readonly ITestOutputHelper _out;
        public DraftModelSpeculativeBench(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void DraftModel_Speculative_BitIdentical_AndSpeedup_OnNovelText()
        {
            if (!File.Exists(TargetGguf) || !File.Exists(Path.Combine(DraftDir, "model.safetensors")))
            {
                _out.WriteLine("missing target gguf or draft safetensors");
                return;
            }

            using var target = CachedLlamaInferenceEngine.LoadGguf(TargetGguf);
            using var draft = SafetensorsLlamaLoader.Load(DraftDir);   // 0.5B, Q4_K by default
            var tok = QwenTokenizer.Load(DraftDir);

            const string text =
                "Explain in two sentences why the sky appears blue during the day and red at sunset.";
            var promptArr = tok.Encode(text, addBos: false);
            const int generate = 80;

            // ── Reference: plain greedy single-token on the target ──
            var refSeq = new List<int>(generate);
            double singleTokPerSec;
            using (var s = target.CreateSession(1024))
            {
                s.Reset(promptArr);
                var g = SamplingOptions.Greedy;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (var i = 0; i < generate; i++) { refSeq.Add(s.GenerateNextToken(in g)); }
                sw.Stop();
                singleTokPerSec = generate / sw.Elapsed.TotalSeconds;
            }
            _out.WriteLine($"single-token target decode: {singleTokPerSec:F2} tok/s");

            // ── Draft-model speculative (greedy), swept over maxDraft to show the trend is structural ──
            foreach (var maxDraft in new[] { 2, 4, 6 })
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
                        for (var c = 0; c < n; c++) { specSeq.Add(committed[c]); history.Add(committed[c]); }
                        steps++;
                    }
                    sw.Stop();
                    specTokPerSec = specSeq.Count / sw.Elapsed.TotalSeconds;
                }

                _out.WriteLine($"maxDraft={maxDraft}: avg {(double)specSeq.Count / steps:F2} tok/step, " +
                               $"draft-spec {specTokPerSec:F2} tok/s, speedup {specTokPerSec / singleTokPerSec:F2}×");

                // Greedy speculative is EXACT — identical token sequence to single-token decode (every maxDraft).
                for (var i = 0; i < generate; i++)
                {
                    Assert.Equal(refSeq[i], specSeq[i]);
                }
            }
        }
    }
}
