// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Selection;

namespace DevOnBike.Overfit.Tests
{
    public sealed class EvolutionaryTournamentSelectionTests
    {
        [Fact]
        public void Constructor_ThrowsOnNonPositiveTournamentSize()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TournamentSelectionOperator(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new TournamentSelectionOperator(-1));
        }

        [Fact]
        public void SelectParent_ThrowsOnEmptyEliteSet()
        {
            var op = new TournamentSelectionOperator(tournamentSize: 3);

            Assert.Throws<ArgumentException>(() => op.SelectParent(ReadOnlySpan<int>.Empty, new Random(1)));
        }

        [Fact]
        public void SelectParent_WithTournamentSizeOne_IsUniformOverElites()
        {
            // Tournament size 1 reduces to a single uniform draw from the elite set,
            // i.e. identical behavior to TruncationSelectionOperator.
            var op = new TournamentSelectionOperator(tournamentSize: 1);
            int[] elites = [10, 20, 30, 40, 50];
            var rng = new Random(42);

            var counts = new int[5];
            const int trials = 10_000;

            for (var i = 0; i < trials; i++)
            {
                var picked = op.SelectParent(elites, rng);
                var pos = Array.IndexOf(elites, picked);
                counts[pos]++;
            }

            // Expected: each elite picked about 2000 times. 3σ binomial tolerance is ~±120.
            for (var i = 0; i < 5; i++)
            {
                Assert.InRange(counts[i], 1700, 2300);
            }
        }

        [Fact]
        public void SelectParent_WithTournamentSizeEqualToElites_StronglyFavorsBest()
        {
            // Tournament selection as implemented here samples with replacement: positions
            // are drawn independently from uniform over eliteIndices. With k = |elites| the
            // odds of at least one draw hitting position 0 are 1 − ((N−1)/N)ᴺ, which is
            // 1 − (4/5)⁵ ≈ 0.672 for N=5. Best-elite is NOT deterministic in this setting.
            // Instead we verify: over many trials, the fittest elite is picked far more
            // often than any other, roughly matching the theoretical hit-rate.
            int[] elites = [10, 20, 30, 40, 50];
            var op = new TournamentSelectionOperator(tournamentSize: elites.Length);
            var rng = new Random(1);

            var counts = new int[elites.Length];
            const int trials = 10_000;

            for (var i = 0; i < trials; i++)
            {
                var picked = op.SelectParent(elites, rng);
                var pos = Array.IndexOf(elites, picked);
                counts[pos]++;
            }

            // Theoretical P(min-draw = i) = ((N−i)/N)ᴺ − ((N−i−1)/N)ᴺ for N=5.
            //   i=0: (5/5)⁵ − (4/5)⁵ = 1 − 0.32768 ≈ 0.672
            //   i=1: (4/5)⁵ − (3/5)⁵ ≈ 0.250
            //   i=2: (3/5)⁵ − (2/5)⁵ ≈ 0.067
            //   i=3: (2/5)⁵ − (1/5)⁵ ≈ 0.010
            //   i=4: (1/5)⁵ ≈ 0.0003
            // With 10k trials and 3σ binomial tolerance:
            Assert.InRange(counts[0], 6500, 6900);
            Assert.InRange(counts[1], 2300, 2700);
            Assert.InRange(counts[2], 500, 850);
            Assert.InRange(counts[3], 50, 200);
        }

        [Fact]
        public void SelectParent_WithVeryLargeTournamentSize_ConvergesToBest()
        {
            // As k → ∞ the probability of at least one draw landing on position 0 tends to 1,
            // so the picked elite converges to the best one with overwhelming probability.
            int[] elites = [10, 20, 30, 40, 50];
            var op = new TournamentSelectionOperator(tournamentSize: 100);
            var rng = new Random(42);

            // P(NOT picking best) = (4/5)^100 ≈ 2e-10. Across 1000 trials, expect zero misses.
            for (var i = 0; i < 1_000; i++)
            {
                Assert.Equal(elites[0], op.SelectParent(elites, rng));
            }
        }

        [Fact]
        public void SelectParent_HigherTournamentSizeFavorsFitter()
        {
            // Intermediate tournament sizes bias selection toward early elite positions
            // (i.e. fitter candidates), with stronger bias for larger k.
            int[] elites = [100, 200, 300, 400, 500, 600, 700, 800];

            var samplesK2 = SampleHeadFraction(new TournamentSelectionOperator(2), elites, trials: 10_000, headSize: 2);
            var samplesK5 = SampleHeadFraction(new TournamentSelectionOperator(5), elites, trials: 10_000, headSize: 2);

            // With k = 2 on 8 elites, the top-2 share should be significantly higher than
            // uniform (2/8 = 0.25). With k = 5, higher still. Exact analytic expectations:
            //   P(top-2 | k=2) = 1 − (6/8)² ≈ 0.4375
            //   P(top-2 | k=5) = 1 − (6/8)⁵ ≈ 0.763
            Assert.InRange(samplesK2, 0.39f, 0.49f);
            Assert.InRange(samplesK5, 0.72f, 0.80f);
            Assert.True(samplesK5 > samplesK2);
        }

        [Fact]
        public void SelectParent_IsAllocationStable()
        {
            var op = new TournamentSelectionOperator(tournamentSize: 3);
            int[] elites = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            var rng = new Random(42);

            // Warmup.
            _ = op.SelectParent(elites, rng);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 10_000; i++)
            {
                _ = op.SelectParent(elites, rng);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated <= 64, $"SelectParent allocated {allocated} bytes.");
        }

        private static float SampleHeadFraction(
            TournamentSelectionOperator op,
            int[] elites,
            int trials,
            int headSize)
        {
            var rng = new Random(2024);
            var head = new HashSet<int>();
            for (var i = 0; i < headSize; i++)
            {
                head.Add(elites[i]);
            }

            var hits = 0;
            for (var i = 0; i < trials; i++)
            {
                var picked = op.SelectParent(elites, rng);
                if (head.Contains(picked))
                {
                    hits++;
                }
            }

            return hits / (float)trials;
        }
    }
}