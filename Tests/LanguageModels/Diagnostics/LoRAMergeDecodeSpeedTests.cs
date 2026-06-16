// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Validates the payoff of LoRA-merge-into-the-fast-engine (ROADMAP Audio #3): a voice-cloned model should
    /// decode at ~preset speed once the adapter is baked in, instead of crawling on the trainable graph. Measures
    /// best-of-N single-stream decode tok/s on the SAME prompt for three paths on the real Orpheus-3B:
    /// <list type="bullet">
    /// <item><b>trainable</b> — the slow autograd graph (<see cref="VoiceCloneTrainer.Generate"/>), the clone path before merge;</item>
    /// <item><b>preset</b> — the base Q4_K_M zero-alloc engine (no fine-tune), the speed ceiling;</item>
    /// <item><b>merged Q8</b> — the fidelity-preserving default (<c>preferQ4K: false</c>): reads ~2× the bytes of
    /// the Q4_K preset, so it trails the ceiling;</item>
    /// <item><b>merged Q4_K</b> — the opt-in fast path (<c>preferQ4K: true</c>): q/k/v + gate/up/down re-quantize to
    /// 4-bit (O stays Q8), reaching preset decode speed but with coarser weights (see <see cref="MergeDivergenceTests"/>
    /// for the fidelity cost).</item>
    /// </list>
    /// [LongFact] — needs C:\orpheus + C:\myvoice\myvoice_v2.adapter.
    /// </summary>
    public sealed class LoRAMergeDecodeSpeedTests
    {
        private const string Orpheus = @"C:\orpheus\orpheus-3b-0.1-ft-q4_k_m.gguf";
        private const string Adapter = @"C:\myvoice\myvoice_v2.adapter";

        private const int WarmTokens = 4;   // absorb prefill + JIT before timing
        private const int TimedTokens = 48; // long enough to amortize per-call overhead
        private const int Runs = 3;         // best-of-3 (plus one untimed warm-up run)

        private readonly ITestOutputHelper _out;
        public LoRAMergeDecodeSpeedTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Merged_Vs_Preset_DecodeTokensPerSecond()
        {
            if (!File.Exists(Orpheus) || !File.Exists(Adapter)) { _out.WriteLine("missing orpheus/adapter"); return; }

            using var trainer = new VoiceCloneTrainer(Orpheus, maxSeqLen: 256, new QLoRAOptions());
            trainer.LoadAdapter(Adapter);

            // One prompt, reused for all three paths (same model/vocab → fair, identical decode work).
            var prompt = OrpheusPrompt.BuildPromptTokens(trainer.Tokenizer, "Hello, this is a decode speed test.", "myvoice");
            _out.WriteLine($"prompt {prompt.Length} tok, warm {WarmTokens}, timed {TimedTokens}, best of {Runs}");

            // ── trainable (slow autograd graph) — the clone path BEFORE merge ──
            var trainableTokS = MeasureTrainable(trainer, prompt);
            _out.WriteLine($"  trainable (autograd graph) : {trainableTokS,6:F2} tok/s");

            // ── preset (base Q4_K_M fast engine, no fine-tune) — the speed ceiling ──
            double presetTokS;
            using (var preset = CachedLlamaInferenceEngine.LoadGguf(Orpheus))
            {
                presetTokS = MeasureEngine(preset, prompt);
            }
            _out.WriteLine($"  preset    (base  Q4_K fast): {presetTokS,6:F2} tok/s");

            // ── merged Q8 (default) + Q4_K (opt-in) — both built from the same trainer ──
            // NOTE: the merged engine borrows the base's embedding/LM-head; disposing it frees those shared handles,
            // so a SECOND BuildMergedEngine in the same process would hit freed weights. We therefore intentionally
            // do NOT dispose the merged engines here (they're released at process exit) — the documented way to
            // measure two merged variants side by side without the double-dispose.
            var mergedQ8 = trainer.BuildMergedEngine(mergeLora: true, preferQ4K: false);
            var mergedQ8TokS = MeasureEngine(mergedQ8, prompt);
            _out.WriteLine($"  merged Q8 (faithful)       : {mergedQ8TokS,6:F2} tok/s");

            // q/k/v + gate/up/down go 4-bit (O stays Q8 — its row length is not a 256-multiple).
            var mergedQ4k = trainer.BuildMergedEngine(mergeLora: true, preferQ4K: true);
            var mergedQ4kTokS = MeasureEngine(mergedQ4k, prompt);
            _out.WriteLine($"  merged Q4_K (faster)       : {mergedQ4kTokS,6:F2} tok/s");

            _out.WriteLine(string.Empty);
            _out.WriteLine($"  Q8   / trainable : {mergedQ8TokS / trainableTokS:F2}×   Q8   / preset : {mergedQ8TokS / presetTokS:F2}×  (fidelity default)");
            _out.WriteLine($"  Q4_K / trainable : {mergedQ4kTokS / trainableTokS:F2}×   Q4_K / preset : {mergedQ4kTokS / presetTokS:F2}×  (~1.0 = matches the Q4_K ceiling)");

            // The whole point of the merge is to escape the trainable graph — assert that at least held for both.
            Assert.True(mergedQ8TokS > trainableTokS && mergedQ4kTokS > trainableTokS,
                $"merge gave no decode speed-up (Q8 {mergedQ8TokS:F2}, Q4_K {mergedQ4kTokS:F2}, trainable {trainableTokS:F2})");
            // Q4_K reads ~half the bytes of Q8 on the wide projections → should be the faster merged variant.
            Assert.True(mergedQ4kTokS > mergedQ8TokS,
                $"Q4_K merge ({mergedQ4kTokS:F2}) was not faster than Q8 merge ({mergedQ8TokS:F2})");
        }

        // Fast-engine decode tok/s: fresh session per run, prefill + warm tokens untimed, then time TimedTokens.
        private static double MeasureEngine(CachedLlamaInferenceEngine eng, int[] prompt)
        {
            var sampling = SamplingOptions.GreedyWithPenalty(1f);
            var best = 0.0;
            for (var run = 0; run <= Runs; run++)
            {
                using var s = eng.CreateSession();
                s.Prefill(prompt);
                for (var i = 0; i < WarmTokens; i++) { s.GenerateNextToken(in sampling); }

                var t0 = Stopwatch.GetTimestamp();
                for (var i = 0; i < TimedTokens; i++) { s.GenerateNextToken(in sampling); }
                var secs = (Stopwatch.GetTimestamp() - t0) / (double)Stopwatch.Frequency;

                if (run > 0) { best = Math.Max(best, TimedTokens / secs); }
            }
            return best;
        }

        // Trainable-graph decode tok/s via Generate (eos unreachable so it emits the full count). Generate re-runs
        // prefill each call; with TimedTokens ≫ prompt it is decode-dominated, comparable to the ROADMAP baseline.
        private static double MeasureTrainable(VoiceCloneTrainer trainer, int[] prompt)
        {
            var best = 0.0;
            for (var run = 0; run <= Runs; run++)
            {
                var t0 = Stopwatch.GetTimestamp();
                _ = trainer.Generate(prompt, TimedTokens, eosTokenId: -999, temperature: 0f, repeatPenalty: 1f, seed: 1);
                var secs = (Stopwatch.GetTimestamp() - t0) / (double)Stopwatch.Frequency;

                if (run > 0) { best = Math.Max(best, TimedTokens / secs); }
            }
            return best;
        }
    }
}
