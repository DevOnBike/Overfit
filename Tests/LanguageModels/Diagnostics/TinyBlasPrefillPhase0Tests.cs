// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Phase 0 for the tinyBLAS register-tiling investigation (docs/ROADMAP perf): MEASURE before building.
    /// A register-tiled MR×NR Q4_K GEMM only pays off where there is a second dimension (N&gt;1 rows), i.e.
    /// PREFILL — never single-token decode GEMV (memory-bound, already at DRAM floor). This test quantifies,
    /// on the real Qwen-3B Q4_K_M, the three go/no-go premises:
    ///
    ///   (A) prefill's share of a realistic session (does the lever matter at all?),
    ///   (B) prefill cost-per-token vs prompt length (falling ⇒ weight-bandwidth-bound ⇒ batching/tiling helps;
    ///       flat ⇒ already compute-bound ⇒ little headroom),
    ///   (C) A/B of the existing weight-stationary lever (re-decode-per-row vs weight-stationary) — the
    ///       current amortisation, a proxy for how much a true 2D tile could still add.
    ///
    /// Emits a table + a verdict line. [LongFact] — needs C:\qwen3b\qwen.q4km.gguf. Run best-of-N (min) so a
    /// noisy box doesn't decide the outcome (canary: the decode number should be stable across sizes).
    /// </summary>
    public sealed class TinyBlasPrefillPhase0Tests
    {
        private const string Path = @"C:\qwen3b\qwen.q4km.gguf";
        private const int Context = 2048;
        private const int DecodeTokens = 48;   // decode sample for per-token timing
        private const int Repeats = 3;         // best-of-N (min) per measurement

        private static readonly int[] PromptLengths = [128, 256, 512, 1024];

        private readonly ITestOutputHelper _out;

        public TinyBlasPrefillPhase0Tests(ITestOutputHelper output) => _out = output;

        [ModelFact(Path, "1min2s")]
        public void Phase0_PrefillHeadroom_AndSplit()
        {

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var sampling = SamplingOptions.Greedy;

            // ── decode per-token baseline (memory-bound; used for the split + as the box canary) ──
            var decodeMsPerTok = MeasureDecode(engine, in sampling);
            _out.WriteLine($"decode: {decodeMsPerTok:F2} ms/token  ({1000.0 / decodeMsPerTok:F1} tok/s)  [memory-bound baseline]");
            _out.WriteLine("");

            // ── prefill scaling + weight-stationary A/B ──
            _out.WriteLine("prompt |  prefill ms (WS on) |  ms/token |  prefill ms (WS off) |  WS speedup");
            _out.WriteLine("-------+---------------------+-----------+----------------------+------------");

            var prefillAtL = new System.Collections.Generic.Dictionary<int, double>();
            foreach (var l in PromptLengths)
            {
                var prompt = SyntheticPrompt(l);

                BatchedQuantProjection.UseWeightStationaryQ4K = true;
                var wsOn = MeasurePrefill(engine, prompt);

                BatchedQuantProjection.UseWeightStationaryQ4K = false;
                var wsOff = MeasurePrefill(engine, prompt);

                BatchedQuantProjection.UseWeightStationaryQ4K = true; // restore default

                prefillAtL[l] = wsOn;
                _out.WriteLine(
                    $"{l,6} | {wsOn,19:F1} | {wsOn / l,9:F3} | {wsOff,20:F1} | {wsOff / wsOn,10:F2}×");
            }

            _out.WriteLine("");

            // ── (A) prefill share of a realistic session: prompt=512, generate=128 ──
            const int genTokens = 128;
            var pf512 = prefillAtL[512];
            var decodeTotal = decodeMsPerTok * genTokens;
            var share = pf512 / (pf512 + decodeTotal);
            _out.WriteLine(
                $"(A) session prompt=512 + gen={genTokens}: prefill {pf512:F0} ms vs decode {decodeTotal:F0} ms " +
                $"→ prefill share = {share * 100:F1}%");

            // ── (B) bandwidth-bound signal: does prefill ms/token fall as the prompt grows? ──
            var perTok128 = prefillAtL[128] / 128.0;
            var perTok1024 = prefillAtL[1024] / 1024.0;
            var amortization = perTok128 / perTok1024;
            _out.WriteLine(
                $"(B) prefill ms/token: {perTok128:F3} @128 → {perTok1024:F3} @1024  " +
                $"(amortization {amortization:F2}×; >1.3 ⇒ weight-bandwidth-bound ⇒ tiling headroom)");

            // ── verdict ──
            _out.WriteLine("");
            var prefillMatters = share >= 0.15;
            var hasHeadroom = amortization >= 1.3;
            var verdict = (prefillMatters, hasHeadroom) switch
            {
                (true, true) => "GO — prefill is a meaningful share AND is bandwidth-bound with headroom for a 2D tile.",
                (true, false) => "MARGINAL — prefill matters but is already compute-bound; a tile may add little. Bench a projection directly before committing.",
                (false, true) => "LOW-PRIORITY — headroom exists but prefill is a small share of real sessions.",
                (false, false) => "STOP — prefill is neither a big share nor bandwidth-bound. Document as premise-fail.",
            };
            _out.WriteLine($"VERDICT: {verdict}");
        }

        private static double MeasureDecode(CachedLlamaInferenceEngine engine, in SamplingOptions sampling)
        {
            var best = double.MaxValue;
            var prompt = SyntheticPrompt(32);
            for (var r = 0; r < Repeats; r++)
            {
                using var session = engine.CreateSession(Context);
                session.Reset(prompt);
                for (var i = 0; i < 4; i++)
                {
                    session.GenerateNextToken(in sampling); // warm
                }

                var sw = Stopwatch.StartNew();
                var n = 0;
                for (; n < DecodeTokens && !session.IsFull; n++)
                {
                    session.GenerateNextToken(in sampling);
                }
                sw.Stop();
                if (n > 0)
                {
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / n);
                }
            }
            return best;
        }

        private static double MeasurePrefill(CachedLlamaInferenceEngine engine, int[] prompt)
        {
            // Warm once (page-in weights + JIT the batched path), then best-of-N on a fresh cache each time.
            using (var warm = engine.CreateSession(Context))
            {
                warm.Reset(prompt);
            }

            var best = double.MaxValue;
            for (var r = 0; r < Repeats; r++)
            {
                using var session = engine.CreateSession(Context);
                var sw = Stopwatch.StartNew();
                session.Reset(prompt); // Reset() clears the cache; Prefill() runs the batched GEMMs
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
        }

        // Deterministic valid token ids (small ids are always < vocab). Timing doesn't need meaningful text.
        private static int[] SyntheticPrompt(int length)
        {
            var ids = new int[length];
            var seed = 0x9E3779B1u;
            for (var i = 0; i < length; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                ids[i] = (int)(seed % 2000u) + 1; // 1..2000, safely inside vocab
            }
            return ids;
        }
    }
}
