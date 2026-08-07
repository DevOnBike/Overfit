// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// The assertion greedy speculative decoding can actually keep: <b>it diverges from single-token greedy
    /// only where the two kernels' own arithmetic difference permits it.</b>
    ///
    /// <para><b>Why the obvious assertion is wrong.</b> Three tests asserted an identical token sequence and
    /// all three failed the first time they were ever run (2026-08-07). The verify forward is batched
    /// (<c>PrefillBatchedQuantAllRows</c> + <c>ProjectLogitsBatched</c>); the reference decodes one token at
    /// a time. Different summation order, and floating-point addition is not associative — measured on
    /// Qwen2.5-3B Q4_K_M, <c>max|Δlogit|</c> between the two paths on an IDENTICAL context is 0.47–1.02,
    /// while the gap between the top two tokens is routinely smaller than that (0.43 after one token). An
    /// argmax flip is therefore not a defect, it is licensed by the arithmetic, and over a long generation
    /// it eventually happens. Bit-identity was never available to assert.</para>
    ///
    /// <para><b>Why there is no threshold constant here.</b> A bound guessed for one model is a bound wrong
    /// for the next, and this repository has already paid for carrying a measured number into a comparison
    /// it did not belong to. So the check MEASURES the kernel difference at the divergence position, on the
    /// model under test, and then asks whether the flip fits inside it. What it rejects is a divergence too
    /// large for arithmetic to explain — which is what a genuine defect in the commit logic would look
    /// like, and what this would have caught had it existed.</para>
    /// </summary>
    internal static class SpeculativeDivergence
    {
        /// <summary>
        /// Doubling the measured difference, for the same reason a bridge is not built to exactly its load:
        /// the difference is sampled at ONE position with ONE batch shape, and the verify's shape
        /// (1 + drafts) is not the prefill's. A flip inside 2× is arithmetic; one outside it is not, and a
        /// commit-logic defect would land far outside — the observed flip used 0.13 of an available 0.54.
        /// </summary>
        private const float ToleranceFactor = 2f;

        /// <summary>
        /// Asserts that <paramref name="speculative"/> matches <paramref name="greedy"/> except where the
        /// batched/single-token logit difference accounts for it, and reports the arithmetic either way.
        /// </summary>
        /// <param name="newSession">Creates a fresh session on the TARGET model; called twice.</param>
        internal static void AssertOnlyNearTieFlips(
            Func<CachedLlamaSession> newSession,
            int[] prompt,
            IReadOnlyList<int> greedy,
            IReadOnlyList<int> speculative,
            int count,
            ITestOutputHelper output)
        {
            var divergence = -1;

            for (var i = 0; i < count; i++)
            {
                if (greedy[i] != speculative[i])
                {
                    divergence = i;

                    break;
                }
            }

            if (divergence < 0)
            {
                output.WriteLine($"speculative == greedy over all {count} tokens.");

                return;
            }

            // Both paths agree on everything before `divergence`, so stepping the reference there reproduces
            // the exact context the speculative step saw.
            float[] singleToken;

            using (var replay = newSession())
            {
                replay.Reset(prompt);
                var sampling = SamplingOptions.Greedy;

                for (var i = 0; i < divergence; i++)
                {
                    replay.GenerateNextToken(in sampling);
                }

                singleToken = new float[replay.VocabularySize];
                replay.GetLastLogits(singleToken);
            }

            // The same context reached by ONE batched prefill — the same class of kernel the verify uses.
            float[] batched;

            using (var prefilled = newSession())
            {
                var context = new int[prompt.Length + divergence];
                prompt.CopyTo(context, 0);

                for (var i = 0; i < divergence; i++)
                {
                    context[prompt.Length + i] = greedy[i];
                }

                prefilled.Reset(context);
                batched = new float[prefilled.VocabularySize];
                prefilled.GetLastLogits(batched);
            }

            var kernelDifference = 0f;

            for (var t = 0; t < singleToken.Length; t++)
            {
                var difference = MathF.Abs(singleToken[t] - batched[t]);

                if (difference > kernelDifference)
                {
                    kernelDifference = difference;
                }
            }

            var ranked = Enumerable.Range(0, singleToken.Length)
                .OrderByDescending(t => singleToken[t])
                .ToArray();
            var chosen = speculative[divergence];
            var rank = Array.IndexOf(ranked, chosen) + 1;
            var deficit = singleToken[ranked[0]] - singleToken[chosen];

            output.WriteLine($"first divergence at index {divergence}: greedy {greedy[divergence]}, "
                + $"speculative {chosen}");
            output.WriteLine($"   single-token ranking of the speculative choice: rank {rank}, "
                + $"deficit {deficit:F6} below the winner");
            output.WriteLine($"   top-2 gap at that position:                     {singleToken[ranked[0]] - singleToken[ranked[1]]:F6}");
            output.WriteLine($"   measured batched-vs-single max|delta logit|:    {kernelDifference:F6}");
            output.WriteLine($"   budget ({ToleranceFactor:F0}x measured):                          {ToleranceFactor * kernelDifference:F6}");

            Assert.True(
                deficit <= ToleranceFactor * kernelDifference,
                $"Speculative decoding picked token {chosen} at index {divergence}, {deficit:F6} below the "
                + $"single-token winner {greedy[divergence]} (rank {rank}). The two kernels differ by at most "
                + $"{kernelDifference:F6} on this context, so a flip of {deficit:F6} is NOT explained by "
                + "summation order — this looks like a defect in the commit logic, not arithmetic.");
        }
    }
}
