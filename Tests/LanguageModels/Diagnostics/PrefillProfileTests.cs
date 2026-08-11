// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Sizes the PREFILL path (time-to-first-token), which <c>DecodeProfiler</c> does not cover — its hooks
    /// live in the single-token decode path only.
    ///
    /// <para>The question this answers: <b>how much of TTFT is tokenization?</b> That is the only thing that
    /// decides whether tokenizer-level work (SearchValues, FrozenDictionary vocab, span-based scanning) can
    /// pay for itself. The decode profile already showed the answer for decode — tokenization does not appear
    /// there at all, because it runs once, before the first token.</para>
    ///
    /// <para>Deliberately coarse: three stopwatches around tokenize / prefill / first-decoded-token. A finer
    /// split would need permanent profiler hooks in the batched-prefill path, which is not worth adding to the
    /// library for a one-off sizing.</para>
    /// </summary>
    public sealed class PrefillProfileTests
    {
        private readonly ITestOutputHelper _out;

        public PrefillProfileTests(ITestOutputHelper output) => _out = output;

        [LongFact("38s")]
        public void Prefill_TokenizeVsForward_Shares()
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

            // A prompt long enough to take the BATCHED prefill path (the short path decodes token-by-token
            // and would measure something else entirely).
            var paragraph = string.Join(" ",
                Enumerable.Repeat(
                    "The history of computing began with mechanical calculators and evolved through vacuum tubes, "
                    + "transistors, integrated circuits and finally the microprocessor era.", 24));

            // Warm-up: JIT, page-in the weights, prime the tokenizer tables.
            {
                using var warm = engine.CreateSession(1024);
                var warmIds = tok.Encode(paragraph);
                warm.Reset(warmIds.AsSpan(0, Math.Min(64, warmIds.Length)));
                warm.GenerateNextToken(in sampling);
            }

            const int Runs = 5;
            var tokenizeMs = new double[Runs];
            var prefillMs = new double[Runs];
            var firstTokenMs = new double[Runs];
            var promptLength = 0;

            for (var r = 0; r < Runs; r++)
            {
                using var session = engine.CreateSession(1024);

                var sw = Stopwatch.StartNew();
                var ids = tok.Encode(paragraph);
                sw.Stop();
                tokenizeMs[r] = sw.Elapsed.TotalMilliseconds;
                promptLength = ids.Length;

                sw.Restart();
                session.Reset(ids);
                sw.Stop();
                prefillMs[r] = sw.Elapsed.TotalMilliseconds;

                sw.Restart();
                session.GenerateNextToken(in sampling);
                sw.Stop();
                firstTokenMs[r] = sw.Elapsed.TotalMilliseconds;
            }

            // Median, not mean: one page-fault or a background task skews a 5-run mean badly.
            static double Median(double[] values)
            {
                var copy = (double[])values.Clone();
                Array.Sort(copy);
                return copy[copy.Length / 2];
            }

            var t = Median(tokenizeMs);
            var p = Median(prefillMs);
            var f = Median(firstTokenMs);
            var ttft = t + p + f;

            _out.WriteLine($"=== Prefill profile ({promptLength} prompt tokens, median of {Runs}) ===");
            _out.WriteLine($"  tokenize      : {t,9:F3} ms   {100.0 * t / ttft,5:F1}%");
            _out.WriteLine($"  prefill fwd   : {p,9:F3} ms   {100.0 * p / ttft,5:F1}%");
            _out.WriteLine($"  first token   : {f,9:F3} ms   {100.0 * f / ttft,5:F1}%");
            _out.WriteLine($"  TTFT total    : {ttft,9:F3} ms");
            _out.WriteLine($"  prefill rate  : {promptLength / (p / 1000.0),9:F0} tok/s");
            _out.WriteLine($"  tokenize rate : {promptLength / (t / 1000.0),9:F0} tok/s");
            _out.WriteLine($"  raw tokenize  : {string.Join(" / ", tokenizeMs.Select(x => x.ToString("F3")))}");
            _out.WriteLine($"  raw prefill   : {string.Join(" / ", prefillMs.Select(x => x.ToString("F3")))}");

            Assert.True(promptLength > 128, $"prompt too short to exercise batched prefill ({promptLength} tokens)");
        }

        /// <summary>
        /// Splits prefill into attention vs FFN via <see cref="PrefillProfiler"/>. This is the measurement
        /// that decides where the 2.34×-at-equal-ISA gap to llama.cpp actually sits: in the FFN matmuls, or
        /// in the attention path where Q and O are dispatched once per head over the same activations.
        /// </summary>
        [LongFact("31s")]
        public void Prefill_ComponentBreakdown()
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
                    + "transistors, integrated circuits and finally the microprocessor era.", 24));
            var ids = tok.Encode(paragraph);

            // Warm up OUTSIDE the profiled region: JIT, page-in, one-off repack.
            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(ids);
                warm.GenerateNextToken(in sampling);
            }

            PrefillProfiler.Reset();
            PrefillProfiler.Enabled = true;
            try
            {
                const int Runs = 3;
                for (var r = 0; r < Runs; r++)
                {
                    using var session = engine.CreateSession(1024);
                    session.Reset(ids);
                }
            }
            finally
            {
                PrefillProfiler.Enabled = false;
            }

            _out.WriteLine(PrefillProfiler.Report());
            Assert.True(PrefillProfiler.Rows > 0, "profiler recorded no prefill rows — hooks not reached");
        }
    }
}
