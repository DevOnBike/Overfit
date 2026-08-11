// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Prefill cost as a function of <b>prompt length</b> — the axis every prefill measurement in this project
    /// has ignored.
    ///
    /// <para><b>Why this exists.</b> A whole optimisation campaign tuned prefill on a 672-token prompt and took
    /// it from 143 to ~299 tok/s. A later comparison against another pure-.NET engine measured our
    /// time-to-first-token on a ~25-token prompt at <b>410 ms against their 66 ms</b>, while per-token decode
    /// was slightly in our favour. 410 ms over 25 tokens is 16 ms per prompt token, against 3.3 ms per token at
    /// 672 — the same engine, five times worse per token, purely because the prompt is short.</para>
    ///
    /// <para>Long prompts are what benchmarks use; short prompts are what interactive chat actually sends. If
    /// per-token cost climbs as the prompt shrinks, the campaign optimised the benchmark rather than the user's
    /// experience, and this test is what would have caught it.</para>
    ///
    /// <para><b>What it separates.</b> Engine-level prefill only — no HTTP, no serialisation, no sampling. Put
    /// next to the 410 ms measured through the server, the difference is per-request overhead rather than
    /// kernel cost, and the two need very different fixes.</para>
    ///
    /// <para>The <c>OVERFIT_PRECOMPUTED_SCALES</c> arm tests the leading suspect: the F16 scale decode is
    /// hoisted to once per projection, and that cost is independent of row count — so it amortises over 672
    /// rows and may not over 16.</para>
    /// </summary>
    public sealed class PrefillLatencyByPromptLengthTests
    {
        private static readonly int[] PromptLengths = [8, 16, 32, 64, 128, 256, 512, 672];

        private const int Repeats = 3;

        private readonly ITestOutputHelper _out;

        public PrefillLatencyByPromptLengthTests(ITestOutputHelper output) => _out = output;

        [LongFact("22s")]
        public void Prefill_CostPerPromptToken_AcrossLengths()
        {
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);
            var sampling = SamplingOptions.Greedy;

            var paragraph = string.Join(" ",
                Enumerable.Repeat(
                    "The history of computing began with mechanical calculators and evolved through vacuum tubes, "
                    + "transistors, integrated circuits and finally the microprocessor era.", 40));
            var allIds = tok.Encode(paragraph);

            // Warm up outside every measurement: JIT, page-in the weights, the one-off repack.
            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(allIds.AsSpan(0, Math.Min(64, allIds.Length)).ToArray());
                warm.GenerateNextToken(in sampling);
            }

            _out.WriteLine($"batched-prefill threshold is 16 tokens; hoisted scales = "
                + $"{(Environment.GetEnvironmentVariable("OVERFIT_PRECOMPUTED_SCALES") != "0" ? "ON" : "OFF")}");
            _out.WriteLine(string.Empty);
            _out.WriteLine($"  {"tokens",7}{"prefill ms",12}{"ms/token",11}{"tok/s",10}");

            foreach (var length in PromptLengths)
            {
                if (length > allIds.Length)
                {
                    continue;
                }

                var ids = allIds.AsSpan(0, length).ToArray();
                var best = double.MaxValue;

                for (var r = 0; r < Repeats; r++)
                {
                    using var session = engine.CreateSession(1024);
                    var started = ValueStopwatch.StartNew();
                    session.Reset(ids);
                    best = Math.Min(best, started.GetElapsedTime().TotalMilliseconds);
                }

                _out.WriteLine(
                    $"  {length,7}{best,10:F1} ms{best / length,10:F2}{length / (best / 1000.0),10:F0}");
            }

            Assert.True(allIds.Length >= 16, "prompt corpus too short to cross the batched threshold");
        }
    }
}
