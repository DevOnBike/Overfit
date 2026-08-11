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
    /// Prices the F16 scale decode inside a real prefill, at a chat-sized prompt.
    ///
    /// <para><b>Why an earlier A/B could not see this.</b> Toggling <c>UsePrecomputedScales</c> compares two
    /// ways of doing the same fixed work — hoisted to once per projection, or inline once per column tile.
    /// Neither arm removes it, so the comparison came out neutral and the cost stayed invisible. Only
    /// <c>AblateF16Scales</c> actually deletes the work, and it is bypassed when the hoist is on (the kernel
    /// then reads a precomputed buffer). Removing the cost therefore needs <b>both</b>: hoist off, ablate on.</para>
    ///
    /// <para><b>The arithmetic that makes it a suspect.</b> A 2.1 GB Q4_K model holds roughly 1.8 million
    /// weight blocks, each carrying 16 F16 scale/min values, and every one is widened to float on every
    /// prefill — about 29 million scalar conversions, <i>independent of prompt length</i>. A prompt-length
    /// sweep fitted prefill at <c>~175 ms + 3.06 ms/token</c>, and dispatch overhead plus the weight walk
    /// account for under 16% of that constant.</para>
    ///
    /// <para><b>Why it matters.</b> At 672 tokens the constant is 8% of prefill; at a chat-sized ~25 tokens it
    /// is about 70%, which is what a user waiting for the first token actually experiences. Results are wrong
    /// while ablated — this measures cost, never correctness.</para>
    /// </summary>
    public sealed class PrefillScaleDecodeCostTests
    {
        private static readonly int[] Lengths = [16, 64, 672];

        private const int Repeats = 3;

        private readonly ITestOutputHelper _out;

        public PrefillScaleDecodeCostTests(ITestOutputHelper output) => _out = output;

        [LongFact("47s")]
        public void Prefill_ScaleDecodeShare_ByPromptLength()
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

            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(allIds.AsSpan(0, 64).ToArray());
                warm.GenerateNextToken(in sampling);
            }

            _out.WriteLine($"  {"tokens",7}{"baseline",12}{"no scales",12}{"scale cost",12}{"share",8}");

            foreach (var length in Lengths)
            {
                var ids = allIds.AsSpan(0, length).ToArray();

                var baseline = Best(engine, ids, ablate: false);
                var ablated = Best(engine, ids, ablate: true);
                var cost = baseline - ablated;

                _out.WriteLine(
                    $"  {length,7}{baseline,9:F1} ms{ablated,9:F1} ms{cost,9:F1} ms{100 * cost / baseline,7:F0}%");
            }

            Assert.True(allIds.Length >= 672, "prompt corpus too short");
        }

        /// <summary>
        /// Short prompts starve the parallel pool: prefill fans out over <b>column tiles</b>, and
        /// <c>tiles = rows / 8</c>, so a 16-token prompt produces <b>two</b> work items for sixteen cores while
        /// a 672-token prompt produces eighty-four. The component breakdown shows every component about 4×
        /// less efficient per token at 16 tokens, which is what two cores instead of ~thirteen looks like.
        ///
        /// <para><c>UseOutputBlocking</c> fans out over <b>output-row bands</b> instead, so the work-item count
        /// stops depending on prompt length. It was built earlier and measured at −20% on a 672-token prompt,
        /// where tiles are plentiful and banding only adds overhead — and was therefore left off. This tests
        /// the case it was never tried on, where the same property that made it useless is exactly what is
        /// missing.</para>
        /// </summary>
        [LongFact("1min13s")]
        public void Prefill_OutputBlocking_OnShortPrompts()
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

            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(allIds.AsSpan(0, 64).ToArray());
                warm.GenerateNextToken(in sampling);
            }

            _out.WriteLine($"  {"tokens",7}{"tiles",7}{"per-tile",12}{"banded",12}{"speedup",10}");

            foreach (var length in new[] { 16, 32, 64, 128, 256, 672 })
            {
                var ids = allIds.AsSpan(0, length).ToArray();

                var perTile = BestWithBanding(engine, ids, banded: false);
                var banded = BestWithBanding(engine, ids, banded: true);

                _out.WriteLine(
                    $"  {length,7}{(length + 7) / 8,7}{perTile,9:F1} ms{banded,9:F1} ms{perTile / banded,9:F2}x");
            }

            Assert.True(allIds.Length >= 672, "prompt corpus too short");
        }

        private static double BestWithBanding(CachedLlamaInferenceEngine engine, int[] ids, bool banded)
        {
            BatchedQuantProjection.UseOutputBlocking = banded;

            try
            {
                var best = double.MaxValue;

                for (var r = 0; r < Repeats; r++)
                {
                    using var session = engine.CreateSession(1024);
                    var started = ValueStopwatch.StartNew();
                    session.Reset(ids);
                    best = Math.Min(best, started.GetElapsedTime().TotalMilliseconds);
                }

                return best;
            }
            finally
            {
                BatchedQuantProjection.UseOutputBlocking = false;
            }
        }

        private static double Best(CachedLlamaInferenceEngine engine, int[] ids, bool ablate)
        {
            // Both switches are needed: the ablation flag only reaches the inline decode, which the hoist
            // bypasses by reading a precomputed buffer.
            BatchedQuantProjection.UsePrecomputedScales = !ablate;
            Q4KGemvKernel.AblateF16Scales = ablate;

            try
            {
                var best = double.MaxValue;

                for (var r = 0; r < Repeats; r++)
                {
                    using var session = engine.CreateSession(1024);
                    var started = ValueStopwatch.StartNew();
                    session.Reset(ids);
                    best = Math.Min(best, started.GetElapsedTime().TotalMilliseconds);
                }

                return best;
            }
            finally
            {
                BatchedQuantProjection.UsePrecomputedScales = true;
                Q4KGemvKernel.AblateF16Scales = false;
            }
        }
    }
}
