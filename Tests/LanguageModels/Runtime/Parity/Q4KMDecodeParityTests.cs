// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Step 3.4 — end-to-end parity of the Q4_K_M decode path (Q4_K + Q6_K
    /// resident, per llama.cpp's "_M" mix) against in-file references, both
    /// loaded from the SAME <c>qwen.q4km.gguf</c>. The loader's <c>quantize</c>
    /// flag toggles between F32-fallback (dequant everything to F32) and the
    /// K-quant native path; <c>GgufLlamaLoader.Load(path)</c> defaults to true.
    ///
    /// Methodology — teacher-forced top-1 parity, mirroring
    /// <see cref="Q8DecodeParityTests"/>:
    ///   1. Reference: greedy-generate N tokens from the canonical prompt;
    ///      record argmax + full logit vector per step.
    ///   2. Subject: re-run the SAME N steps feeding the reference token at
    ///      every step (teacher forcing) so a single argmax flip cannot
    ///      cascade — each step measures Q4_K + Q6_K quantization error in
    ///      isolation.
    ///   3. Characterise every mismatch by the reference's logit margin over
    ///      the subject's pick (the "swing").
    ///
    /// Q4_K is 4-bit (vs Q8's 8-bit), so per-projection quantization noise is
    /// inherently larger than Q8_0's. The bar is therefore "flips only at
    /// genuine near-ties", not "zero flips" — same shape as the Q8 parity test
    /// but with looser thresholds documented from measurement.
    ///
    /// <para>
    /// One omission worth naming: there's no "Q4_K_M file vs FP16-quantized
    /// engine" test, despite Q8 having an analogous one. The reason is that
    /// Q4_K_M and Q8 are <i>two different quantizations of the same base model</i>
    /// — not the same model decoded two ways. They genuinely disagree on
    /// confident tokens (measured: step 0 had a 10.88-logit F32-margin gap),
    /// which would be flagged as a "bug" by the swing-threshold test but is
    /// in fact the inherent Q4 vs Q8 model gap. Correctness is established by
    /// the F32-baseline test below (same-file dequant vs Q4_K + Q6_K resident).
    /// </para>
    ///
    /// Two engines load sequentially so peak RAM stays at one model's footprint.
    ///
    /// <see cref="LongFactAttribute"/>: loads the real 3B model — skipped by default.
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Quantization")]
    [Trait("Category", "Parity")]
    public sealed class Q4KMDecodeParityTests
    {
        // Canonical prompt shared with the rest of the Qwen suite: [BOS, im_start, "\n"].
        private static readonly int[] Prompt = [151643, 151644, 198];

        private const int Steps = 32;
        private const int MaxContext = 64;

        private readonly ITestOutputHelper _out;

        public Q4KMDecodeParityTests(ITestOutputHelper output)
        {
            _out = output;
        }

        /// <summary>
        /// Step 3.4 — Q4_K_M decode (FFN gate/up + attn Q/K = Q4_K, FFN down +
        /// attn V + token_embd + output = Q6_K, attn_output = dequant→Q8) vs an
        /// F32 reference loaded from the same Q4_K_M file. Isolates the
        /// Q4_K+Q6_K decode noise from any file-vs-file dequantization difference.
        /// </summary>
        [LongFact]
        public void Q4KMDecodePath_TopTokenMatches_F32Baseline_TeacherForced()
        {
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();

            // ── Phase 1 — F32 reference (quantization disabled) ────────────────
            int[] refTokens;
            float[][] refLogits;
            {
                using var f32 = GgufLlamaLoader.Load(path, quantize: false);
                (refTokens, refLogits) = Run(f32, Steps, forceTokens: null);
            }   // F32 engine disposed before the Q4_K_M engine loads.

            // ── Phase 2 — Q4_K + Q6_K decode path, teacher-forced along refTokens ──
            int[] subjArgmax;
            float[][] subjLogits;
            {
                using var subj = GgufLlamaLoader.Load(path);   // quantize: true (default)
                (subjArgmax, subjLogits) = Run(subj, Steps, forceTokens: refTokens);
            }

            var (matches, worstSwing) = ReportParity(
                "Q4_K_M decode path vs F32 baseline (same Q4_K_M file)",
                refTokens, refLogits, subjArgmax, subjLogits);

            // Q4_K_M is 4-bit FFN + 6-bit boost tensors; per-projection noise
            // is larger than Q8's. The bar is "flips only at genuine near-ties".
            // Measured (Qwen2.5-3B Q4_K_M, 32 steps): 29/32 top-1 match, worst
            // mismatch swing 2.16 — better than the Q8-vs-F32 test's 28/32 (the
            // F32 reference here is *dequant-from-Q4_K_M*, so the reference
            // noise floor matches what Q4_K_M-resident sees).
            Assert.True(matches >= 22,
                $"Q4_K_M top-1 matched F32 at only {matches}/{Steps} — more divergence " +
                "than Q4_K + Q6_K quantization noise explains (measured: 29/32).");

            // Real correctness check: every mismatch is a genuine near-tie.
            // Q4_K is coarser than Q8 (4-bit vs 8-bit), but the swing stayed
            // comparable to Q8's (Q8: worst 1.39, Q4_K: worst 2.16). Bar set
            // 2.3× the measured worst.
            Assert.True(worstSwing < 5.0f,
                $"A mismatch had swing {worstSwing:F3} ≥ 5.0 — Q4_K_M picked a token " +
                "F32 ranked far lower; that is a decode error, not Q4_K noise.");
        }

        /// <summary>
        /// Per-step parity analysis + report. Returns (top-1 match count, worst
        /// mismatch swing). swing = ref-margin + subj-margin: how far apart the
        /// two engines rank the contested token pair (small = genuine near-tie).
        /// </summary>
        private (int matches, float worstSwing) ReportParity(
            string label,
            int[] refTokens,
            float[][] refLogits,
            int[] subjArgmax,
            float[][] subjLogits)
        {
            var steps = refTokens.Length;
            var matches = 0;
            var maxHeadDiff = 0f;
            var worstSwing = 0f;
            var mismatches = new List<string>();

            for (var i = 0; i < steps; i++)
            {
                var refTok = refTokens[i];
                var subjTok = subjArgmax[i];
                var rf = refLogits[i];
                var sf = subjLogits[i];

                var headDiff = MathF.Abs(rf[refTok] - sf[refTok]);
                if (headDiff > maxHeadDiff) { maxHeadDiff = headDiff; }

                if (subjTok == refTok)
                {
                    matches++;
                }
                else
                {
                    var refMargin = rf[refTok] - rf[subjTok];
                    var subjMargin = sf[subjTok] - sf[refTok];
                    var swing = refMargin + subjMargin;
                    if (swing > worstSwing) { worstSwing = swing; }
                    mismatches.Add(
                        $"step {i,2}: ref→{refTok} subj→{subjTok}  " +
                        $"ref-margin={refMargin:F3}  subj-margin={subjMargin:F3}  swing={swing:F3}");
                }
            }

            _out.WriteLine($"{label} — {steps} teacher-forced steps:");
            _out.WriteLine($"  top-1 match           = {matches}/{steps}");
            _out.WriteLine($"  max |Δlogit| at top-1 = {maxHeadDiff:F4}  (incl. benign uniform offsets)");
            _out.WriteLine($"  worst mismatch swing  = {worstSwing:F4}");
            foreach (var m in mismatches)
            {
                _out.WriteLine($"  {m}");
            }

            return (matches, worstSwing);
        }

        /// <summary>
        /// Greedy-decodes <paramref name="steps"/> tokens from <see cref="Prompt"/>.
        /// When <paramref name="forceTokens"/> is null the session advances by its
        /// own argmax (free greedy generation); otherwise it advances by the
        /// supplied reference token (teacher forcing). Returns each step's argmax
        /// token and a copy of the full logit vector.
        /// </summary>
        private static (int[] argmax, float[][] logits) Run(
            CachedLlamaInferenceEngine engine, int steps, int[]? forceTokens)
        {
            using var session = engine.CreateSession(MaxContext);
            session.Reset(Prompt);

            var argmax = new int[steps];
            var logits = new float[steps][];

            for (var i = 0; i < steps; i++)
            {
                var step = session.LastLogits.ToArray();
                logits[i] = step;
                argmax[i] = ArgMax(step);

                var feed = forceTokens is null ? argmax[i] : forceTokens[i];
                session.Prefill([feed]);
            }

            return (argmax, logits);
        }

        private static int ArgMax(ReadOnlySpan<float> values)
        {
            var bestIdx = 0;
            var bestVal = values[0];
            for (var i = 1; i < values.Length; i++)
            {
                if (values[i] > bestVal)
                {
                    bestVal = values[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }
    }
}
