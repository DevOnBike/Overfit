// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Sampling
{
    /// <summary>
    /// The partial-sort optimisation for top-p / typical-p must be EXACT: the survivor set must be identical
    /// whether the candidate cap forces (a) a full sort (cap ≥ vocab), (b) the partial path + full-sort
    /// fallback (tiny cap), or (c) the normal partial-hit path (default cap). Verified via
    /// <see cref="TokenSampler.ComputeProbabilities"/> (survivors = non-zero probability).
    /// </summary>
    public sealed class TopPTypicalPartialSortParityTests
    {
        private const int Vocab = 2000;

        [Theory]
        [InlineData(true)]  // top-p
        [InlineData(false)] // typical-p
        public void PartialSort_MatchesFullSort_AcrossCapSizes(bool topP)
        {
            // A peaked distribution → the surviving set sits between the tiny cap (forces fallback) and the
            // default cap (partial-hit), so all three code paths are exercised on the same input.
            var logits = new float[Vocab];
            var rng = new Random(123);
            for (var i = 0; i < Vocab; i++)
            {
                logits[i] = (float)(rng.NextDouble() * 2.0);
            }
            for (var i = 0; i < 40; i++)
            {
                logits[rng.Next(Vocab)] += 6f; // a handful of clear peaks
            }

            var opts = topP
                ? new SamplingOptions(SamplingStrategy.TopP, 0.8f, 0, 0.9f, 0)
                : SamplingOptions.WithTypicalP(0.9f, 0.8f);

            var fullSort = SurvivorSet(logits, opts, capOverride: Vocab);   // full sort
            var fallback = SurvivorSet(logits, opts, capOverride: 4);       // partial + full-sort fallback
            var partialHit = SurvivorSet(logits, opts, capOverride: 1024);  // normal partial-hit path

            Assert.True(fullSort.Count > 4, "test distribution should keep more than the tiny cap");
            Assert.Equal(fullSort, fallback);
            Assert.Equal(fullSort, partialHit);
        }

        private static SortedSet<int> SurvivorSet(float[] logits, SamplingOptions opts, int capOverride)
        {
            var saved = TokenSampler.NucleusPartialCap;
            TokenSampler.NucleusPartialCap = capOverride;
            try
            {
                var idx = new int[logits.Length];
                var score = new float[logits.Length];
                var probs = new float[logits.Length];
                TokenSampler.ComputeProbabilities(logits, in opts, idx, score, probs);

                var survivors = new SortedSet<int>();
                for (var i = 0; i < probs.Length; i++)
                {
                    if (probs[i] > 0f)
                    {
                        survivors.Add(i);
                    }
                }
                return survivors;
            }
            finally
            {
                TokenSampler.NucleusPartialCap = saved;
            }
        }
    }
}
