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
    /// Step 2.5 — end-to-end parity of the Q8_0 decode path against an F32
    /// reference, both loaded from the SAME qwen.gguf. <see cref="GgufLlamaLoader"/>'s
    /// <c>quantize</c> flag toggles whether FFN / LM-head / attention weights are
    /// quantized on load, so the only difference between the two engines is the
    /// decode-weight precision.
    ///
    /// Methodology — teacher-forced top-1 parity:
    ///   1. F32 reference: greedy-generate N tokens from the canonical prompt;
    ///      record each step's argmax token + full logit vector.
    ///   2. Q8 path: re-run the SAME N steps, but feed the F32 reference token
    ///      at every step (teacher forcing) so both models always see an
    ///      identical prefix. A single argmax flip therefore cannot cascade —
    ///      each step measures Q8 quantization error in isolation.
    ///   3. Compare per step: Q8 argmax vs F32 argmax, and characterise every
    ///      mismatch by how strongly F32 preferred its own pick over Q8's.
    ///
    /// Q8_0 decode is lossy by construction — 8-bit weights *and* 8-bit
    /// activations. The correctness bar is therefore not "no flips" but "flips
    /// only at genuine near-ties": at a mismatch F32's logit margin between its
    /// pick and Q8's pick must be small (Q8 chose a close runner-up, not a token
    /// F32 strongly rejected). A layout / dequant bug instead shows up as Q8
    /// picking tokens with a large F32 margin against them.
    ///
    /// The engines load sequentially (F32 ~14 GB disposed before Q8 ~6 GB loads)
    /// so peak RAM stays at the F32 footprint, not the sum.
    ///
    /// Measured on Qwen2.5-3B (dev box, 32 steps): 28/32 teacher-forced top-1
    /// match; worst mismatch swing 1.39; all 4 mismatches at F32 margins ≤ 0.47
    /// — genuine near-ties (e.g. "get" vs "create", "using" vs "looking"). The
    /// generated text is coherent and near-identical to F32. Verdict: the Q8
    /// decode path is correct (lossy, as Q8_0 inherently is).
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

        [LongFact]
        public void Q8DecodePath_TopTokenMatches_F32Baseline_TeacherForced()
        {
            var path = TestModelPaths.Qwen3B.GgufPath;
            TestModelPaths.Qwen3B.RequireGgufPath();

            // ── Phase 1 — F32 reference (quantization disabled) ────────────────
            int[] refTokens;
            float[][] refLogits;
            {
                using var f32 = GgufLlamaLoader.Load(path, quantize: false);
                (refTokens, refLogits) = Run(f32, Steps, forceTokens: null);
            }   // F32 engine (~14 GB) disposed before the Q8 engine loads.

            // ── Phase 2 — Q8 decode path, teacher-forced along refTokens ───────
            int[] q8Argmax;
            float[][] q8Logits;
            {
                using var q8 = GgufLlamaLoader.Load(path);   // quantize: true (default)
                (q8Argmax, q8Logits) = Run(q8, Steps, forceTokens: refTokens);
            }

            // ── Per-step parity analysis ───────────────────────────────────────
            var matches = 0;
            var maxHeadDiff = 0f;     // max |Δlogit| at the F32 top-1 token (decides argmax)
            var worstSwing = 0f;      // largest model disagreement at a mismatch
            var mismatches = new List<string>();

            for (var i = 0; i < Steps; i++)
            {
                var refTok = refTokens[i];
                var q8Tok = q8Argmax[i];
                var rf = refLogits[i];
                var qf = q8Logits[i];

                var headDiff = MathF.Abs(rf[refTok] - qf[refTok]);
                if (headDiff > maxHeadDiff) { maxHeadDiff = headDiff; }

                if (q8Tok == refTok)
                {
                    matches++;
                }
                else
                {
                    // F32's margin between its pick and Q8's pick (≥ 0 — refTok is
                    // F32's argmax); Q8's margin the other way. A small swing =
                    // both models nearly tied = a genuine ambiguous position.
                    var f32Margin = rf[refTok] - rf[q8Tok];
                    var q8Margin = qf[q8Tok] - qf[refTok];
                    var swing = f32Margin + q8Margin;
                    if (swing > worstSwing) { worstSwing = swing; }
                    mismatches.Add(
                        $"step {i,2}: F32→{refTok} Q8→{q8Tok}  " +
                        $"F32-margin={f32Margin:F3}  Q8-margin={q8Margin:F3}  swing={swing:F3}");
                }
            }

            _out.WriteLine($"Teacher-forced Q8-vs-F32 parity over {Steps} steps:");
            _out.WriteLine($"  top-1 match            = {matches}/{Steps}");
            _out.WriteLine($"  max |Δlogit| at top-1  = {maxHeadDiff:F4}");
            _out.WriteLine($"  worst mismatch swing   = {worstSwing:F4}");
            _out.WriteLine($"  ref tokens = [{string.Join(", ", refTokens)}]");
            _out.WriteLine($"  q8  argmax = [{string.Join(", ", q8Argmax)}]");
            foreach (var m in mismatches)
            {
                _out.WriteLine($"  {m}");
            }

            // ── Assertions ─────────────────────────────────────────────────────
            // Q8_0 decode is lossy (8-bit weights AND activations); the bar is
            // "flips only at genuine near-ties", not "zero flips". Measured on the
            // dev box (Qwen2.5-3B, 32 steps): 28/32 match, worst swing 1.39, every
            // mismatch at an F32 margin ≤ 0.47 — all genuine near-ties.

            // (1) Gross-breakage guard: a working Q8_0 decode tracks F32 closely;
            //     a bad layout/dequant would diverge nearly everywhere.
            Assert.True(matches >= 26,
                $"Q8 top-1 matched F32 at only {matches}/{Steps} — far more " +
                "divergence than Q8_0 quantization noise explains (measured: 28/32).");

            // (2) The real correctness check: every mismatch must be a genuine
            //     near-tie. swing = F32-margin + Q8-margin — how far apart the two
            //     models rank the contested pair. A near-tie has a small swing; a
            //     structural bug makes Q8 pick a token F32 strongly rejected, which
            //     blows the swing up. Measured worst swing: 1.39.
            Assert.True(worstSwing < 3.0f,
                $"A mismatch had swing {worstSwing:F3} ≥ 3.0 — Q8 picked a token " +
                "F32 ranked far lower; that is a decode error, not Q8 noise.");

            // maxHeadDiff (absolute |Δlogit| at the F32 top-1 token) is reported
            // but NOT asserted: it includes near-uniform per-step logit offsets,
            // which softmax and argmax are both invariant to. A large value there
            // is benign as long as the swing (relative ranking) stays small.
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
