// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Parity for the quantized batched prefill: prefilling a prompt through the batched path
    /// (<see cref="CachedGptStack.PrefillBatchedQuant"/>) must leave the session in the same state as
    /// the single-token loop — i.e. the end-of-prompt logits agree. Verified on the real Qwen2.5-3B
    /// Q4_K_M (RMSNorm + RoPE + GQA 16:2 + SwiGLU + mixed K-quant), the exact path the optimisation
    /// targets. [LongFact].
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Parity")]
    public sealed class BatchedPrefillParityTests
    {
        private const string ModelPath = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public BatchedPrefillParityTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void BatchedPrefill_MatchesSingleToken_OnRealQwen()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            // A ≥16-token prompt to trigger the batched path; arbitrary in-vocab ids.
            var prompt = new int[40];
            for (var i = 0; i < prompt.Length; i++) { prompt[i] = 100 + i * 37; }

            // Batched (default eligibility kicks in for this length).
            using var batched = engine.CreateSession(256);
            batched.Reset(prompt);
            var bLogits = batched.LastLogits.ToArray();

            // Single-token reference (force the loop via the test hook).
            using var single = engine.CreateSession(256);
            single.DisableBatchedPrefillForParity = true;
            single.Reset(prompt);
            var sLogits = single.LastLogits;

            var maxDiff = 0f;
            int argB = 0, argS = 0;
            for (var i = 0; i < sLogits.Length; i++)
            {
                maxDiff = MathF.Max(maxDiff, MathF.Abs(bLogits[i] - sLogits[i]));
                if (bLogits[i] > bLogits[argB]) { argB = i; }
                if (sLogits[i] > sLogits[argS]) { argS = i; }
            }

            _out.WriteLine($"argmax batched={argB} single={argS}  maxAbsLogitDiff={maxDiff:G4}");

            // Same predicted token, and logits agree to within accumulated FP noise across 36 layers.
            Assert.Equal(argS, argB);
            Assert.True(maxDiff < 1e-2f, $"batched vs single logit divergence {maxDiff:G4} (> 1e-2).");
        }

        [LongFact]
        public void BatchedPrefill_TtftSpeedup_OnRealQwen()
        {
            if (!File.Exists(ModelPath)) { _out.WriteLine($"missing {ModelPath}"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(ModelPath);

            var prompt = new int[256];
            for (var i = 0; i < prompt.Length; i++) { prompt[i] = 100 + i * 11; }

            double Time(bool disableBatched)
            {
                var best = double.MaxValue;
                for (var r = 0; r < 3; r++)
                {
                    using var s = engine.CreateSession(512);
                    s.DisableBatchedPrefillForParity = disableBatched;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    s.Reset(prompt);   // prefill the whole prompt (TTFT)
                    sw.Stop();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                }
                return best;
            }

            var single = Time(disableBatched: true);
            var batched = Time(disableBatched: false);
            _out.WriteLine($"TTFT {prompt.Length}-token prompt: single={single:F1} ms  batched={batched:F1} ms  speedup={single / batched:F2}×");
        }
    }
}
