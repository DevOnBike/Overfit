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
    /// Parity of the Q8_0 decode path, two ways:
    ///
    ///   • step 2.5 — Q8 (weights quantized on load from an FP16 GGUF) vs an
    ///     F32 reference loaded from the SAME file (`quantize: false`).
    ///   • step 2.4 — Q8_0-native-file (FFN/LM-head blocks read straight from a
    ///     Q8_0 GGUF, no F32 round-trip) vs the FP16-quantized-on-load engine.
    ///
    /// Methodology — teacher-forced top-1 parity:
    ///   1. Reference: greedy-generate N tokens from the canonical prompt;
    ///      record each step's argmax token + full logit vector.
    ///   2. Subject: re-run the SAME N steps fed the reference token at every
    ///      step (teacher forcing) so a single argmax flip cannot cascade —
    ///      each step measures quantization error in isolation.
    ///   3. Characterise every mismatch by how strongly the reference preferred
    ///      its own pick over the subject's (the "swing").
    ///
    /// Q8_0 decode is lossy by construction (8-bit weights and activations);
    /// the bar is "flips only at genuine near-ties", not "zero flips". A
    /// layout / dequant bug instead makes the subject pick tokens the reference
    /// strongly rejected — a large swing.
    ///
    /// Engines load sequentially so peak RAM stays at one model's footprint.
    ///
    /// <see cref="LongFactAttribute"/>: loads the real 3B model — skipped by default.
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Quantization")]
    [Trait("Category", "Parity")]
    public sealed class Q8DecodeParityTests
    {
        // Canonical prompt shared with the rest of the Qwen suite: [BOS, im_start, "\n"].
        private static readonly int[] Prompt = [151643, 151644, 198];

        private const int Steps = 32;
        private const int MaxContext = 64;

        private readonly ITestOutputHelper _out;

        public Q8DecodeParityTests(ITestOutputHelper output)
        {
            _out = output;
        }

        /// <summary>
        /// Step 2.5 — Q8 decode (weights quantized on load) vs an F32 reference,
        /// both from the same FP16 qwen.gguf. Measured (Qwen2.5-3B, 32 steps):
        /// 28/32 top-1 match, worst swing 1.39 — all flips at genuine near-ties.
        /// </summary>
        [LongFact]
        public void Q8DecodePath_TopTokenMatches_F32Baseline_TeacherForced()
        {
            var path = TestModelPaths.Qwen3B.GgufPath;
            TestModelPaths.Qwen3B.RequireGgufPath();

            int[] refTokens;
            float[][] refLogits;
            {
                using var f32 = GgufLlamaLoader.Load(path, quantize: false);
                (refTokens, refLogits) = Run(f32, Steps, forceTokens: null);
            }   // F32 engine (~14 GB) disposed before the Q8 engine loads.

            int[] q8Argmax;
            float[][] q8Logits;
            {
                using var q8 = GgufLlamaLoader.Load(path);   // quantize: true (default)
                (q8Argmax, q8Logits) = Run(q8, Steps, forceTokens: refTokens);
            }

            var (matches, worstSwing) = ReportParity(
                "Q8-quantized-on-load vs F32 baseline (same FP16 file)",
                refTokens, refLogits, q8Argmax, q8Logits);

            // Q8_0 is lossy; the bar is "flips only at genuine near-ties". A small
            // allowance covers ambiguous positions where the F32 top-1/top-2 gap
            // is within Q8 noise. Measured: 28/32 match, worst swing 1.39.
            Assert.True(matches >= 26,
                $"Q8 top-1 matched F32 at only {matches}/{Steps} — far more " +
                "divergence than Q8_0 quantization noise explains (measured: 28/32).");

            // The real correctness check: every mismatch is a genuine near-tie.
            Assert.True(worstSwing < 3.0f,
                $"A mismatch had swing {worstSwing:F3} ≥ 3.0 — Q8 picked a token " +
                "F32 ranked far lower; that is a decode error, not Q8 noise.");
        }

        /// <summary>
        /// Step 2.4 — the Q8_0-native-file load path: FFN / LM-head weights read
        /// straight from a Q8_0 GGUF (no F32 round-trip / re-quantize). Verifies
        /// it decodes like the FP16-quantized-on-load engine. Both are Q8
        /// approximations of the same model, just derived differently (llama.cpp's
        /// Q8 of the original weights vs our Q8 of the FP16 weights), so they
        /// track even closer than the F32-vs-Q8 test above. Measured on the dev
        /// box (Qwen2.5-3B, 32 steps): 32/32 top-1 match, worst swing 0.00.
        /// </summary>
        [LongFact]
        public void Q8_0NativeFile_DecodesLike_Fp16QuantizedOnLoad_TeacherForced()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();
            TestModelPaths.Qwen3B.RequireQ8GgufPath();

            // Reference — FP16 GGUF, weights quantized to Q8 on load.
            int[] refTokens;
            float[][] refLogits;
            {
                using var fp16 = GgufLlamaLoader.Load(TestModelPaths.Qwen3B.GgufPath);
                (refTokens, refLogits) = Run(fp16, Steps, forceTokens: null);
            }

            // Subject — Q8_0 GGUF, FFN/LM-head blocks read natively (step 2.4).
            int[] q8Argmax;
            float[][] q8Logits;
            {
                using var q8file = GgufLlamaLoader.Load(TestModelPaths.Qwen3B.Q8GgufPath);
                (q8Argmax, q8Logits) = Run(q8file, Steps, forceTokens: refTokens);
            }

            var (matches, worstSwing) = ReportParity(
                "Q8_0-native-file vs FP16-quantized-on-load",
                refTokens, refLogits, q8Argmax, q8Logits);

            // Two Q8 renderings of the same model — should track very closely;
            // any mismatch must be a genuine near-tie, never a strong disagreement.
            Assert.True(matches >= 30,
                $"Q8_0-file decode matched the FP16-quantized engine at only " +
                $"{matches}/{Steps} — more divergence than two Q8 renderings explain " +
                "(measured: 32/32).");

            Assert.True(worstSwing < 3.0f,
                $"A mismatch had swing {worstSwing:F3} ≥ 3.0 — the Q8_0-native read " +
                "picked a strongly-rejected token; likely a block de-interleave bug.");
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
