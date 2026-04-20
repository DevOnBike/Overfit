// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Mutation;
using DevOnBike.Overfit.Evolutionary.Selection;
using DevOnBike.Overfit.Evolutionary.Storage;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    ///     Tests for the "best-ever" semantics introduced in round 10: after a strong
    ///     generation sets a high-water mark, a later weaker generation must not demote
    ///     <c>BestFitness</c> / <c>GetBestParameters()</c>.
    /// </summary>
    public sealed class BestEverSemanticsTests
    {
        // ------------------------------------------------------------------
        // GA
        // ------------------------------------------------------------------

        [Fact]
        public void Ga_BestFitness_DoesNotRegressAfterWeakerGeneration()
        {
            // First generation: strong fitness [10, 20, ..., 80]. Best = 80.
            // Second generation: uniformly weak fitness [0, 0, ..., 0]. Best-of-last-gen = 0.
            // New contract: BestFitness must stay at 80 (the high-water mark), not drop to 0.
            using var ga = new GenerationalGeneticAlgorithm(
                populationSize: 8,
                parameterCount: 2,
                eliteFraction: 0.5f,
                selectionOperator: new UniformEliteParentSelector(),
                mutationOperator: new IdentityMutationOperator(),
                fitnessShaper: null,
                seed: 42);

            ga.Initialize(min: 1f, max: 1f); // all weights set to 1 so genome is unambiguous

            var pop = new float[8 * 2];
            ga.Ask(pop);

            float[] strong = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f];
            ga.Tell(strong);

            Assert.Equal(80f, ga.BestFitness, 5);

            // Snapshot best params from the strong generation.
            var bestAfterStrong = ga.GetBestParameters().ToArray();

            // Weak generation.
            ga.Ask(pop);
            float[] weak = [0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f];
            ga.Tell(weak);

            // Best-ever must persist.
            Assert.Equal(80f, ga.BestFitness, 5);
            Assert.Equal(bestAfterStrong, ga.GetBestParameters().ToArray());
        }

        [Fact]
        public void Ga_BestFitness_UpgradesOnImprovement()
        {
            using var ga = new GenerationalGeneticAlgorithm(
                populationSize: 4,
                parameterCount: 2,
                eliteFraction: 0.5f,
                selectionOperator: new UniformEliteParentSelector(),
                mutationOperator: new IdentityMutationOperator(),
                fitnessShaper: null,
                seed: 42);

            ga.Initialize();
            var pop = new float[4 * 2];

            ga.Ask(pop);
            ga.Tell([5f, 10f, 1f, 2f]); // best = 10
            Assert.Equal(10f, ga.BestFitness, 5);

            ga.Ask(pop);
            ga.Tell([100f, 50f, 25f, 1f]); // best = 100
            Assert.Equal(100f, ga.BestFitness, 5);
        }

        [Fact]
        public void Ga_BestFitness_IgnoresAllNaNGeneration()
        {
            // A NaN-only generation after a good one must not corrupt the tracked best.
            using var ga = new GenerationalGeneticAlgorithm(
                populationSize: 4,
                parameterCount: 2,
                eliteFraction: 0.5f,
                selectionOperator: new UniformEliteParentSelector(),
                mutationOperator: new IdentityMutationOperator(),
                fitnessShaper: null,
                seed: 42);

            ga.Initialize();
            var pop = new float[4 * 2];

            ga.Ask(pop);
            ga.Tell([3f, 5f, 4f, 2f]);
            var bestBefore = ga.GetBestParameters().ToArray();
            Assert.Equal(5f, ga.BestFitness, 5);

            ga.Ask(pop);
            ga.Tell([float.NaN, float.NaN, float.NaN, float.NaN]);

            // NaN ranking still produces *some* elite index, so Tell does not throw; the
            // best-ever guard must prevent the NaN from overwriting the tracked best.
            Assert.Equal(5f, ga.BestFitness, 5);
            Assert.Equal(bestBefore, ga.GetBestParameters().ToArray());
        }

        // ------------------------------------------------------------------
        // ES
        // ------------------------------------------------------------------

        [Fact]
        public void Es_BestFitness_DoesNotRegressAfterWeakerGeneration()
        {
            var noise = new PrecomputedNoiseTable(length: 1 << 12, seed: 11);
            using var es = new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise,
                seed: 42);

            es.Initialize();

            var pop = new float[8 * 4];
            es.Ask(pop);
            es.Tell([1f, 2f, 3f, 50f, 5f, 6f, 7f, 8f]); // best = 50

            var bestGenomeAfterStrong = es.GetBestParameters().ToArray();
            Assert.Equal(50f, es.BestFitness, 5);

            es.Ask(pop);
            es.Tell([0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f]); // best-of-current = 0

            Assert.Equal(50f, es.BestFitness, 5);
            Assert.Equal(bestGenomeAfterStrong, es.GetBestParameters().ToArray());
        }

        [Fact]
        public void Es_BestFitness_IgnoresAllNaNGeneration()
        {
            // Old behavior set BestFitness to NaN after an all-NaN generation. New contract:
            // keep the previously recorded best untouched.
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise,
                seed: 42);

            es.Initialize();

            var pop = new float[8 * 4];
            es.Ask(pop);
            es.Tell([1f, 2f, 3f, 10f, 5f, 6f, 7f, 8f]);
            var bestBefore = es.GetBestParameters().ToArray();
            Assert.Equal(10f, es.BestFitness, 5);

            es.Ask(pop);
            var allNaN = new float[8];
            Array.Fill(allNaN, float.NaN);
            es.Tell(allNaN);

            Assert.Equal(10f, es.BestFitness, 5);
            Assert.Equal(bestBefore, es.GetBestParameters().ToArray());
        }

        [Fact]
        public void Es_BestFitness_UpgradesOnImprovement()
        {
            var noise = new PrecomputedNoiseTable(length: 1024, seed: 1);
            using var es = new OpenAiEsStrategy(
                populationSize: 8,
                parameterCount: 4,
                sigma: 0.1f,
                learningRate: 0.01f,
                noiseTable: noise,
                seed: 42);

            es.Initialize();
            var pop = new float[8 * 4];

            es.Ask(pop);
            es.Tell([5f, 10f, 1f, 2f, 3f, 4f, 6f, 7f]);
            Assert.Equal(10f, es.BestFitness, 5);

            es.Ask(pop);
            es.Tell([100f, 50f, 25f, 1f, 2f, 3f, 4f, 5f]);
            Assert.Equal(100f, es.BestFitness, 5);
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        /// <summary>
        ///     Mutation that simply copies parent to child. Makes GA tests deterministic by
        ///     removing the mutation-induced genome drift from the picture.
        /// </summary>
        private sealed class IdentityMutationOperator : IMutationOperator
        {
            public void Mutate(ReadOnlySpan<float> parentGenome, Span<float> childGenome, Random rng)
            {
                parentGenome.CopyTo(childGenome);
            }
        }
    }
}
