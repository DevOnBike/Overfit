// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Attributes the <b>fixed cost of a prefill call</b> — the part that does not scale with prompt length.
    ///
    /// <para><b>The number being explained.</b> A prompt-length sweep fitted prefill as
    /// <c>~175 ms + 3.06 ms per token</c>: the constant is 8% of a 672-token prefill and about 70% of a
    /// chat-sized one. Every optimisation in the prefill campaign moved the per-token term. Nothing has ever
    /// touched the constant, and for interactive use the constant is most of the latency.</para>
    ///
    /// <para><b>Two candidates, measured rather than argued:</b></para>
    /// <list type="number">
    ///   <item><b>Parallel launch overhead.</b> A prefill issues one fan-out per projection per layer. If the
    ///     count is in the hundreds and each launch costs hundreds of microseconds, that alone is the
    ///     constant. Counted with <c>OverfitParallel.CountDispatches</c>, priced with an empty-body loop.</item>
    ///   <item><b>The weight walk.</b> Prefill must read every weight once no matter how short the prompt, so
    ///     there is a floor of <c>model bytes ÷ bandwidth</c> that no compute optimisation can remove. If that
    ///     floor is most of the constant, short prefill is memory-bound exactly like decode — and the same
    ///     conclusion applies: a wider kernel cannot help.</item>
    /// </list>
    /// </summary>
    public sealed class PrefillFixedCostTests
    {
        private const int ShortPrompt = 16;
        private const int LongPrompt = 672;

        private readonly ITestOutputHelper _out;

        public PrefillFixedCostTests(ITestOutputHelper output) => _out = output;

        [LongFact("5s")]
        public unsafe void Prefill_FixedCost_DispatchesVersusWeightWalk()
        {
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            var modelBytes = new FileInfo(path).Length;

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);
            var sampling = SamplingOptions.Greedy;

            var paragraph = string.Join(" ",
                Enumerable.Repeat(
                    "The history of computing began with mechanical calculators and evolved through vacuum tubes, "
                    + "transistors, integrated circuits and finally the microprocessor era.", 40));
            var allIds = tok.Encode(paragraph);

            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(allIds.AsSpan(0, 64).ToArray());
                warm.GenerateNextToken(in sampling);
            }

            // ── 1. How many fan-outs does one prefill issue, and does the count depend on prompt length? ──
            var shortDispatches = CountDispatches(engine, allIds, ShortPrompt);
            var longDispatches = CountDispatches(engine, allIds, LongPrompt);

            _out.WriteLine($"dispatches per prefill: {ShortPrompt,4} tokens -> {shortDispatches,6}");
            _out.WriteLine($"                        {LongPrompt,4} tokens -> {longDispatches,6}");

            // ── 2. What does one fan-out cost when the body does nothing? ──
            var perDispatchUs = MeasureEmptyDispatchMicroseconds();
            var launchMs = shortDispatches * perDispatchUs / 1000.0;

            _out.WriteLine(string.Empty);
            _out.WriteLine($"empty fan-out cost      : {perDispatchUs,8:F1} us");
            _out.WriteLine($"launch cost at {ShortPrompt} tokens: {launchMs,8:F1} ms"
                + $"  ({100 * launchMs / 175.0,5:F0}% of the ~175 ms constant)");

            // ── 3. The floor: every weight must be read once regardless of prompt length. ──
            var readGbps = MeasureReadGigabytesPerSecond();
            var walkMs = modelBytes / (readGbps * 1e9) * 1000.0;

            _out.WriteLine(string.Empty);
            _out.WriteLine($"model on disk           : {modelBytes / 1e9,8:F2} GB");
            _out.WriteLine($"measured read bandwidth : {readGbps,8:F1} GB/s");
            _out.WriteLine($"weight-walk floor       : {walkMs,8:F1} ms"
                + $"  ({100 * walkMs / 175.0,5:F0}% of the ~175 ms constant)");

            _out.WriteLine(string.Empty);
            _out.WriteLine($"accounted               : {100 * (launchMs + walkMs) / 175.0,5:F0}% of the constant");

            Assert.True(shortDispatches > 0, "no dispatches counted — the counter hook was not reached");
        }

        private static long CountDispatches(CachedLlamaInferenceEngine engine, int[] allIds, int length)
        {
            var ids = allIds.AsSpan(0, length).ToArray();

            using var session = engine.CreateSession(1024);

            OverfitParallel.ResetDispatchCount();
            OverfitParallel.CountDispatches = true;
            try
            {
                session.Reset(ids);
            }
            finally
            {
                OverfitParallel.CountDispatches = false;
            }

            return OverfitParallel.DispatchCount;
        }

        /// <summary>
        /// Cost of one fan-out with a body that does nothing — pure launch, wake and join. The range is wide
        /// enough that the pool actually fans out rather than taking the inline fast path, which is the whole
        /// point: the inline path is free and is not what a prefill pays.
        /// </summary>
        private static unsafe double MeasureEmptyDispatchMicroseconds()
        {
            const int Iterations = 2000;

            // Warm the pool so thread wake-up is not charged to the first samples.
            for (var i = 0; i < 100; i++)
            {
                OverfitParallel.For(0, 1024, 1, &NoOp, null);
            }

            var best = double.MaxValue;

            for (var r = 0; r < 5; r++)
            {
                var started = ValueStopwatch.StartNew();

                for (var i = 0; i < Iterations; i++)
                {
                    OverfitParallel.For(0, 1024, 1, &NoOp, null);
                }

                best = Math.Min(best, started.GetElapsedTime().TotalMilliseconds);
            }

            return best * 1000.0 / Iterations;
        }

        private static unsafe void NoOp(int start, int end, void* context)
        {
        }

        /// <summary>Single-core sequential read rate, the rate a weight walk can realistically achieve.</summary>
        private static double MeasureReadGigabytesPerSecond()
        {
            const int Floats = 64 * 1024 * 1024; // 256 MB — past any cache

            var data = new float[Floats];
            for (var i = 0; i < Floats; i++)
            {
                data[i] = i;
            }

            var best = double.MaxValue;

            for (var r = 0; r < 3; r++)
            {
                var started = ValueStopwatch.StartNew();
                var total = 0f;

                for (var i = 0; i < Floats; i += 8)
                {
                    total += data[i];
                }

                var elapsed = started.GetElapsedTime().TotalSeconds;
                GC.KeepAlive(total);
                best = Math.Min(best, elapsed);
            }

            return (double)Floats * sizeof(float) / best / 1e9;
        }
    }
}
