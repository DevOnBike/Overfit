// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Crossover;

namespace DevOnBike.Overfit.Tests
{
    public sealed class EvolutionarySbxCrossoverTests
    {
        [Fact]
        public void Constructor_ThrowsOnNegativeDistributionIndex()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SbxCrossoverOperator(distributionIndex: -1f));
        }

        [Fact]
        public void Constructor_ThrowsOnProbabilityOutsideUnitInterval()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SbxCrossoverOperator(perGeneProbability: -0.1f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new SbxCrossoverOperator(perGeneProbability: 1.1f));
        }

        [Fact]
        public void Crossover_ThrowsOnMismatchedGenomeLengths()
        {
            var op = new SbxCrossoverOperator();

            Assert.Throws<ArgumentException>(() => op.Crossover(
                new float[4],
                new float[4],
                new float[4],
                new float[3],
                new Random(1)));
        }

        [Fact]
        public void Crossover_WithProbabilityZero_CopiesParentsThrough()
        {
            // p = 0 => every gene skips SBX => children == parents.
            var op = new SbxCrossoverOperator(perGeneProbability: 0f);

            float[] p1 = [1f, 2f, 3f, 4f];
            float[] p2 = [5f, 6f, 7f, 8f];
            var c1 = new float[4];
            var c2 = new float[4];

            op.Crossover(p1, p2, c1, c2, new Random(1));

            Assert.Equal(p1, c1);
            Assert.Equal(p2, c2);
        }

        [Fact]
        public void Crossover_PreservesMidpointOfParents()
        {
            // Core SBX invariant: for every gene that crosses over,
            //   child1[i] + child2[i] == parent1[i] + parent2[i]
            // holds exactly by construction, regardless of β. Any per-gene skip also
            // preserves the sum (skip = identity copy). So the sum equality holds unconditionally.
            var op = new SbxCrossoverOperator(distributionIndex: 15f, perGeneProbability: 1f);

            float[] p1 = [-2f, 0.5f, 3f, 10f];
            float[] p2 = [4f, -1.5f, 0f, -5f];
            var c1 = new float[4];
            var c2 = new float[4];
            var rng = new Random(42);

            for (var trial = 0; trial < 100; trial++)
            {
                op.Crossover(p1, p2, c1, c2, rng);

                for (var i = 0; i < p1.Length; i++)
                {
                    var parentSum = p1[i] + p2[i];
                    var childSum = c1[i] + c2[i];
                    Assert.Equal(parentSum, childSum, 4);
                }
            }
        }

        [Fact]
        public void Crossover_WithIdenticalParents_ProducesIdenticalChildren()
        {
            // If p1 == p2, the SBX formula collapses algebraically to child = p1 regardless
            // of β: 0.5 · ((1+β)·p + (1−β)·p) = p. The identity holds mathematically, but
            // the floating-point evaluation accumulates 1–2 ULPs of rounding error through
            // the (1+β)/(1-β)/·/+ chain, so bit-exact equality is too strict. Assert
            // numerical equality to 5 decimal digits — tight enough to catch any real bug,
            // loose enough to tolerate IEEE 754 rounding.
            var op = new SbxCrossoverOperator();

            float[] p = [1f, -2f, 3.14f, -0.5f, 100f];
            var c1 = new float[p.Length];
            var c2 = new float[p.Length];

            op.Crossover(p, p, c1, c2, new Random(1));

            for (var i = 0; i < p.Length; i++)
            {
                Assert.Equal(p[i], c1[i], 5);
                Assert.Equal(p[i], c2[i], 5);
            }
        }

        [Fact]
        public void Crossover_LargeDistributionIndex_KeepsChildrenNearParents()
        {
            // η → ∞ means β → 1, so children ≈ parents. Concretely: at η = 100, the mean
            // child-to-parent deviation should be small relative to the parent-to-parent
            // distance. Measured across many trials to stabilise the statistic.
            var op = new SbxCrossoverOperator(distributionIndex: 100f);

            float[] p1 = [0f, 0f, 0f, 0f];
            float[] p2 = [10f, 10f, 10f, 10f];
            var c1 = new float[4];
            var c2 = new float[4];
            var rng = new Random(42);

            var totalDrift = 0.0;
            const int trials = 1_000;

            for (var trial = 0; trial < trials; trial++)
            {
                op.Crossover(p1, p2, c1, c2, rng);

                for (var i = 0; i < 4; i++)
                {
                    // Each child should be close to ITS corresponding parent.
                    totalDrift += Math.Abs(c1[i] - p1[i]);
                    totalDrift += Math.Abs(c2[i] - p2[i]);
                }
            }

            var meanDrift = totalDrift / (trials * 4 * 2);

            // Parent-to-parent distance per dimension is 10. Under η = 100, empirical mean
            // drift should be well below 1.0 (< 10% of the parent gap). Proving this from
            // first principles requires integrating the β-distribution and is not a unit
            // test's job; the empirical bound with 1000 trials is a strong smoke test.
            Assert.True(meanDrift < 1.0, $"Expected mean drift < 1.0 for η=100, got {meanDrift:F4}.");
        }

        [Fact]
        public void Crossover_SmallDistributionIndex_ProducesDispersedChildren()
        {
            // Complement of the previous test: η = 2 should produce substantial dispersion.
            var op = new SbxCrossoverOperator(distributionIndex: 2f);

            float[] p1 = [0f, 0f, 0f, 0f];
            float[] p2 = [10f, 10f, 10f, 10f];
            var c1 = new float[4];
            var c2 = new float[4];
            var rng = new Random(42);

            var totalDrift = 0.0;
            const int trials = 1_000;

            for (var trial = 0; trial < trials; trial++)
            {
                op.Crossover(p1, p2, c1, c2, rng);

                for (var i = 0; i < 4; i++)
                {
                    totalDrift += Math.Abs(c1[i] - p1[i]);
                    totalDrift += Math.Abs(c2[i] - p2[i]);
                }
            }

            var meanDrift = totalDrift / (trials * 4 * 2);

            // η = 2 is well below the exploitation regime. Drift per dimension should be
            // comfortably above 1.0.
            Assert.True(meanDrift > 1.5, $"Expected mean drift > 1.5 for η=2, got {meanDrift:F4}.");
        }

        [Fact]
        public void Crossover_IsAllocationStable()
        {
            var op = new SbxCrossoverOperator();

            var p1 = new float[64];
            var p2 = new float[64];
            var c1 = new float[64];
            var c2 = new float[64];
            var rng = new Random(1);

            for (var i = 0; i < 64; i++)
            {
                p1[i] = rng.NextSingle();
                p2[i] = rng.NextSingle();
            }

            // Warmup.
            op.Crossover(p1, p2, c1, c2, rng);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 5_000; i++)
            {
                op.Crossover(p1, p2, c1, c2, rng);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated <= 64, $"Crossover allocated {allocated} bytes.");
        }
    }
}