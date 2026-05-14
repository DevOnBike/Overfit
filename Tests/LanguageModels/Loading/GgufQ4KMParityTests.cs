// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Integration parity test for Q4_K_M dequantization against the FP16 GGUF baseline.
    /// Synthetic unit tests (see <c>GgmlDequantTests</c>) cover the algorithm; this test
    /// catches bit-layout regressions vs llama.cpp/Ollama on a real model.
    ///
    /// To run locally:
    ///   1. Download a Q4_K_M variant of the same model that lives at <c>qwen.gguf</c>:
    ///        ollama pull qwen2.5:3b-instruct-q4_K_M
    ///        ollama show --modelfile qwen2.5:3b-instruct-q4_K_M  # locate blob
    ///        cp &lt;blob&gt; c:\qwen3b\qwen.q4km.gguf
    ///      Or from HuggingFace:
    ///        huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
    ///          qwen2.5-3b-instruct-q4_k_m.gguf --local-dir c:\qwen3b --local-dir-use-symlinks False
    ///        mv c:\qwen3b\qwen2.5-3b-instruct-q4_k_m.gguf c:\qwen3b\qwen.q4km.gguf
    ///   2. Run:  dotnet test -c Release --filter "FullyQualifiedName~GgufQ4KMParityTests"
    ///
    /// Tolerance rationale:
    ///   - Top-1 token MUST match. Greedy decoding is robust to small logit shifts;
    ///     if Q4_K_M selects a different top-1, the dequant or decode is wrong.
    ///   - Top-10 overlap should be ≥ 7. Q4_K_M typically perturbs the tail more than
    ///     the head; the head 10 are dominated by the actual signal.
    ///   - Max abs logit diff is reported but not asserted strictly: real Q4_K_M error
    ///     varies by prompt and layer depth. A diff &gt; 5.0 across the top-10 indices
    ///     would indicate something structurally wrong, not just lossy quant.
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Quantization")]
    [Trait("Category", "Parity")]
    public sealed class GgufQ4KMParityTests
    {
        private static string Fp16ModelPath => TestModelPaths.Qwen3B.GgufPath;
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
        public void Q4KM_TopTokenMatches_FP16Baseline_OnCanonicalPrompt()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();

            // ── Baseline: FP16 GGUF ────────────────────────────────────────────
            var (fp16Top1, fp16Top10, fp16Logits) = RunOneStep(Fp16ModelPath);
            _out.WriteLine($"FP16   top-1 = {fp16Top1}  (logit={fp16Logits[fp16Top1]:F4})");
            _out.WriteLine($"FP16   top-10 = [{string.Join(", ", fp16Top10)}]");

            // ── Subject: Q4_K_M GGUF (same model, lower precision) ─────────────
            var (q4Top1, q4Top10, q4Logits) = RunOneStep(Q4KMModelPath);
            _out.WriteLine($"Q4_K_M top-1 = {q4Top1}  (logit={q4Logits[q4Top1]:F4})");
            _out.WriteLine($"Q4_K_M top-10 = [{string.Join(", ", q4Top10)}]");

            // ── Diff report ────────────────────────────────────────────────────
            var topOverlap = CountOverlap(fp16Top10, q4Top10);
            var maxDiffOverTop10 = MaxAbsDiffAtIndices(fp16Logits, q4Logits, fp16Top10);
            _out.WriteLine($"Top-10 overlap = {topOverlap}/10");
            _out.WriteLine($"Max abs logit diff over FP16 top-10 indices = {maxDiffOverTop10:F4}");

            // ── Hard parity assertions ─────────────────────────────────────────
            Assert.True(fp16Top1 == q4Top1,
                $"Greedy top-1 mismatch: FP16={fp16Top1} vs Q4_K_M={q4Top1}. " +
                "Q4_K_M dequant or block layout is likely wrong.");

            Assert.True(topOverlap >= 7,
                $"Top-10 overlap too low ({topOverlap}/10). Expected ≥ 7 — " +
                "Q4_K_M is lossy but should retain the head distribution.");

            // Sanity guard against catastrophically wrong dequant (e.g. byte-order bug):
            // a working Q4_K_M will be off by ~0.1–1.5 logit units on top tokens.
            Assert.True(maxDiffOverTop10 < 5.0f,
                $"Max abs diff {maxDiffOverTop10:F4} on FP16 top-10 indices > 5.0 — " +
                "structural decode error, not lossy quant noise.");
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// Loads the model at <paramref name="path"/>, primes a 64-context session
        /// with the canonical prompt, and returns the resulting logits + their top
        /// indices. Engine is disposed before returning so successive calls don't
        /// double-stack RAM.
        /// </summary>
        private static (int top1, int[] top10, float[] logits) RunOneStep(string path)
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
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
            for (var i = 0; i < k; i++) { idx[i] = -1; val[i] = float.NegativeInfinity; }

            for (var i = 0; i < logits.Length; i++)
            {
                var v = logits[i];
                // Insert into sorted-descending top-k if v beats the current minimum.
                if (v <= val[k - 1]) { continue; }
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
                    if (a[i] == b[j]) { n++; break; }
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
                if (d > max) { max = d; }
            }
            return max;
        }
    }
}
