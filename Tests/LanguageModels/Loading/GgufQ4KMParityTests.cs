// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Integration parity test for the native Q4_K_M decode path against the F32 dequantization
    /// of the <b>same file</b>. Synthetic unit tests (see <c>GgmlDequantTests</c>) cover the
    /// algorithm; this test catches bit-layout / per-head-split regressions on a real model.
    ///
    /// IMPORTANT — baseline choice: the reference is the <b>F32 dequant of the Q4_K_M file</b>
    /// (<c>quantize: false</c>), NOT the separate FP16 model. Q4_K_M is genuinely lossy: even a
    /// lossless F32 dequant of it shifts the top-1 token on a near-tie prompt relative to the FP16
    /// model, so demanding native-Q4_K_M == FP16 top-1 is unsatisfiable through no fault of the
    /// decode. The same-file comparison isolates the only thing under test here — that the resident
    /// Q4_K/Q6_K kernels + per-head split reproduce the file's own weights (a bit-layout bug blows it
    /// up; legitimate Q8_K-activation noise stays a near-tie). See <c>Q4KMDecodeParityTests</c> for
    /// the multi-step teacher-forced version of the same comparison.
    ///
    /// To run locally:
    ///   1. Download a Q4_K_M variant to <c>c:\qwen3b\qwen.q4km.gguf</c>:
    ///        huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
    ///          qwen2.5-3b-instruct-q4_k_m.gguf --local-dir c:\qwen3b --local-dir-use-symlinks False
    ///        mv c:\qwen3b\qwen2.5-3b-instruct-q4_k_m.gguf c:\qwen3b\qwen.q4km.gguf
    ///   2. Run:  dotnet test -c Release --filter "FullyQualifiedName~GgufQ4KMParityTests"
    ///
    /// Tolerance rationale:
    ///   - Top-1 must match the F32-dequant baseline OR the swap is a genuine near-tie
    ///     (combined logit swing &lt; 3.0). Q8_K activation quantization perturbs near-ties.
    ///   - Top-10 overlap ≥ 7 (the head distribution is dominated by the actual signal).
    ///   - Max abs logit diff over the baseline top-10 &lt; 5.0 — a larger diff is a structural
    ///     decode/layout error, not lossy-quant noise.
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Quantization")]
    [Trait("Category", "Parity")]
    public sealed class GgufQ4KMParityTests
    {
        private static string Q4KMModelPath => TestModelPaths.Qwen3B.Q4KmGgufPath;

        // Canonical 3-token prompt used by the rest of the Qwen test suite:
        //   [BOS, im_start, "\n"] — exercises embedding + first layer non-trivially.
        private static readonly int[] Prompt = [151643, 151644, 198];

        private readonly ITestOutputHelper _out;

        public GgufQ4KMParityTests(ITestOutputHelper output)
        {
            _out = output;
        }

        [LongFact]
        public void Q4KM_TopTokenMatches_F32DequantBaseline_OnCanonicalPrompt()
        {
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();

            // ── Baseline: F32 dequant of the SAME Q4_K_M file (quantize: false) ──
            var (refTop1, refTop10, refLogits) = RunOneStep(Q4KMModelPath, quantize: false);
            _out.WriteLine($"F32deq top-1 = {refTop1}  (logit={refLogits[refTop1]:F4})");
            _out.WriteLine($"F32deq top-10 = [{string.Join(", ", refTop10)}]");

            // ── Subject: native Q4_K / Q6_K resident decode (quantize: true) ────
            var (q4Top1, q4Top10, q4Logits) = RunOneStep(Q4KMModelPath, quantize: true);
            _out.WriteLine($"Q4_K_M top-1 = {q4Top1}  (logit={q4Logits[q4Top1]:F4})");
            _out.WriteLine($"Q4_K_M top-10 = [{string.Join(", ", q4Top10)}]");

            // ── Diff report ────────────────────────────────────────────────────
            var topOverlap = CountOverlap(refTop10, q4Top10);
            var maxDiffOverTop10 = MaxAbsDiffAtIndices(refLogits, q4Logits, refTop10);
            _out.WriteLine($"Top-10 overlap = {topOverlap}/10");
            _out.WriteLine($"Max abs logit diff over F32deq top-10 indices = {maxDiffOverTop10:F4}");

            // ── Parity assertions ──────────────────────────────────────────────
            // Top-1 matches the F32-dequant baseline, or the swap is a genuine near-tie
            // (combined logit swing small — Q8_K activation quantization perturbs near-ties).
            if (refTop1 != q4Top1)
            {
                var swing = (refLogits[refTop1] - refLogits[q4Top1]) + (q4Logits[q4Top1] - q4Logits[refTop1]);
                _out.WriteLine($"Top-1 differs ({refTop1} vs {q4Top1}) — swing={swing:F4}");
                Assert.True(swing < 3.0f,
                    $"Top-1 mismatch swing {swing:F3} ≥ 3.0 — native picked a token F32-dequant ranked " +
                    "far lower; a bit-layout / per-head-split error, not Q8_K-activation noise.");
            }

            Assert.True(topOverlap >= 7,
                $"Top-10 overlap too low ({topOverlap}/10). Expected ≥ 7 — " +
                "Q8_K activation noise perturbs the tail but should retain the head distribution.");

            // Sanity guard against catastrophically wrong dequant (e.g. byte-order bug):
            // a working native path tracks its own F32 dequant to a few logit units on top tokens.
            Assert.True(maxDiffOverTop10 < 5.0f,
                $"Max abs diff {maxDiffOverTop10:F4} on F32deq top-10 indices > 5.0 — " +
                "structural decode error, not Q8_K-activation noise.");
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// Loads the model at <paramref name="path"/>, primes a 64-context session
        /// with the canonical prompt, and returns the resulting logits + their top
        /// indices. Engine is disposed before returning so successive calls don't
        /// double-stack RAM.
        /// </summary>
        private static (int top1, int[] top10, float[] logits) RunOneStep(string path, bool quantize)
        {
            using var engine = GgufLlamaLoader.Load(path, quantize: quantize);
            using var session = engine.CreateSession(64);
            session.Reset(Prompt);

            var logits = session.LastLogits.ToArray();
            var top10 = TopKIndices(logits, 10);
            return (top10[0], top10, logits);
        }

        private static int[] TopKIndices(ReadOnlySpan<float> logits, int k)
        {
            // Simple partial selection — k is tiny (10), vocab is ~150k, no LINQ.
            var idx = new int[k];
            var val = new float[k];
            for (var i = 0; i < k; i++)
            {
                idx[i] = -1;
                val[i] = float.NegativeInfinity;
            }

            for (var i = 0; i < logits.Length; i++)
            {
                var v = logits[i];
                // Insert into sorted-descending top-k if v beats the current minimum.
                if (v <= val[k - 1])
                {
                    continue;
                }
                var pos = k - 1;
                while (pos > 0 && val[pos - 1] < v)
                {
                    val[pos] = val[pos - 1];
                    idx[pos] = idx[pos - 1];
                    pos--;
                }
                val[pos] = v;
                idx[pos] = i;
            }
            return idx;
        }

        private static int CountOverlap(int[] a, int[] b)
        {
            var n = 0;
            for (var i = 0; i < a.Length; i++)
            {
                for (var j = 0; j < b.Length; j++)
                {
                    if (a[i] == b[j])
                    {
                        n++;
                        break;
                    }
                }
            }
            return n;
        }

        private static float MaxAbsDiffAtIndices(float[] a, float[] b, int[] indices)
        {
            var max = 0f;
            for (var i = 0; i < indices.Length; i++)
            {
                var d = MathF.Abs(a[indices[i]] - b[indices[i]]);
                if (d > max)
                {
                    max = d;
                }
            }
            return max;
        }
    }
}
