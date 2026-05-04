// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace DevOnBike.Overfit.Tests
{
    public sealed class EvolutionaryGenerationalGeneticAlgorithmTests
    {
        [Fact]
        public void Ask_ReturnsCurrentPopulationMatrix()
        {
            using var algorithm = CreateAlgorithm(
            selection: new FirstEliteSelectionOperator(),
            mutation: new CopyMutationOperator());

            algorithm.Initialize(min: -0.25f, max: 0.25f);

            var population = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(population);

            Assert.Equal(algorithm.PopulationSize * algorithm.ParameterCount, population.Length);
            Assert.Contains(population, static value => value is >= -0.25f and <= 0.25f);
        }

        [Fact]
        public void Tell_IncrementsGeneration_AndStoresBestFitness()
        {
            using var algorithm = CreateAlgorithm(
            selection: new FirstEliteSelectionOperator(),
            mutation: new CopyMutationOperator());

            algorithm.Initialize();

            var firstPopulation = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(firstPopulation);

            float[] fitness = [1f, 10f, 4f, 3f];
            algorithm.Tell(fitness);

            Assert.Equal(1, algorithm.Generation);
            Assert.Equal(10f, algorithm.BestFitness, 5);

            var best = algorithm.GetBestParameters().ToArray();
            var expected = SliceGenome(firstPopulation, bestGenomeIndex: 1, algorithm.ParameterCount);

            Assert.Equal(expected, best);
        }

        [Fact]
        public void Tell_ProducesChildrenFromElitePool()
        {
            using var algorithm = CreateAlgorithm(
            selection: new FirstEliteSelectionOperator(),
            mutation: new CopyMutationOperator(),
            populationSize: 4,
            parameterCount: 3,
            eliteFraction: 0.5f);

            algorithm.Initialize();

            var before = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(before);

            algorithm.Tell([100f, 90f, 1f, 0f]);

            var after = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(after);

            var elite0 = SliceGenome(before, 0, algorithm.ParameterCount);
            var elite1 = SliceGenome(before, 1, algorithm.ParameterCount);

            for (var genomeIndex = 0; genomeIndex < algorithm.PopulationSize; genomeIndex++)
            {
                var genome = SliceGenome(after, genomeIndex, algorithm.ParameterCount);
                Assert.True(
                genome.SequenceEqual(elite0) || genome.SequenceEqual(elite1),
                $"Genome at index {genomeIndex} should be copied from elite pool.");
            }
        }

        [Fact]
        public void Tell_WithDeterministicMutation_ChangesOnlyChildren()
        {
            using var algorithm = CreateAlgorithm(
            selection: new FirstEliteSelectionOperator(),
            mutation: new AddOneMutationOperator(),
            populationSize: 4,
            parameterCount: 2,
            eliteFraction: 0.5f);

            algorithm.Initialize(min: 0f, max: 0f);

            algorithm.Tell([4f, 3f, 2f, 1f]);

            var after = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(after);

            Assert.Equal([0f, 0f], SliceGenome(after, 0, 2));
            Assert.Equal([0f, 0f], SliceGenome(after, 1, 2));
            Assert.Equal([1f, 1f], SliceGenome(after, 2, 2));
            Assert.Equal([1f, 1f], SliceGenome(after, 3, 2));
        }

        [Fact]
        public void Ask_IsAllocationStable_AfterWarmup()
        {
            using var algorithm = CreateAlgorithm(
            selection: new FirstEliteSelectionOperator(),
            mutation: new CopyMutationOperator());

            algorithm.Initialize();

            var buffer = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(buffer); // warmup

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 10_000; i++)
            {
                algorithm.Ask(buffer);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.True(allocated <= 128, $"Ask allocated {allocated} bytes.");
        }

        [Fact]
        public void Tell_IsAllocationStable_WithNoAllocOperators()
        {
            using var algorithm = CreateAlgorithm(
                selection: new FirstEliteSelectionOperator(),
                mutation: new CopyMutationOperator());

            algorithm.Initialize();

            var fitness = new float[]
            {
        4f,
        3f,
        2f,
        1f
            };

            algorithm.Tell(fitness);

            // warmup
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 5_000; i++)
            {
                algorithm.Tell(fitness);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.True(
                allocated <= 512,
                $"Tell allocated {allocated} bytes.");
        }

        [Fact]
        public void Tell_HandlesNaNFitness_WithoutCorruptingEliteSelection()
        {
            // Regression guard: prior implementation used an insertion sort over the raw
            // fitness array. Comparing NaN with <, > or < returns false, which made the
            // ranking non-deterministic and could place a NaN individual in the elite set.
            // The new PartialSort uses float.CompareTo which treats NaN as smallest,
            // so any valid fitness beats NaN and NaN can never win an elite slot.
            using var algorithm = CreateAlgorithm(
            selection: new FirstEliteSelectionOperator(),
            mutation: new CopyMutationOperator(),
            populationSize: 4,
            parameterCount: 3,
            eliteFraction: 0.5f);

            algorithm.Initialize();

            // Two NaN values and two valid values. The best must be 10f at index 1.
            algorithm.Tell([float.NaN, 10f, float.NaN, 5f]);

            Assert.Equal(10f, algorithm.BestFitness, 5);
        }

        [Fact]
        public void Tell_HandlesAllNaNFitness_WithoutCrashing()
        {
            // Edge case: every individual produces NaN. The algorithm should still make
            // progress (e.g. select an arbitrary individual as "best") rather than crash
            // or loop forever.
            using var algorithm = CreateAlgorithm(
            selection: new FirstEliteSelectionOperator(),
            mutation: new CopyMutationOperator(),
            populationSize: 4,
            parameterCount: 3,
            eliteFraction: 0.5f);

            algorithm.Initialize();

            algorithm.Tell([float.NaN, float.NaN, float.NaN, float.NaN]);

            Assert.Equal(1, algorithm.Generation);
            Assert.True(float.IsNaN(algorithm.BestFitness));
        }

        [Fact]
        public void Tell_LargePopulation_TriggersParallelPath_ProducesValidOffspring()
        {
            // With eliteFraction = 1/256 we get exactly ONE elite (the individual with
            // the highest fitness). Child work = (256 - 1) * 64 = 16320, above the 4096
            // threshold, so the parallel child-creation path is exercised.
            //
            // With a single elite:
            //   - CopyElites writes new[0] = old[bestIndex]
            //   - CreateChildren (parallel) writes new[1..255], each a copy of old[bestIndex]
            //     (FirstEliteSelectionOperator always returns eliteIndices[0] = bestIndex;
            //      CopyMutationOperator reproduces the parent verbatim)
            // So EVERY slot in the new population must equal old[bestIndex]. Any lost
            // update, stale read, or partitioning bug in Parallel.For would leave at least
            // one slot pointing at a different genome (or at zeros).
            using var algorithm = new GenerationalGeneticAlgorithm(
                populationSize: 256,
                parameterCount: 64,
                eliteFraction: 1f / 256f,
                selectionOperator: new FirstEliteSelectionOperator(),
                mutationOperator: new CopyMutationOperator(),
                fitnessShaper: null,
                seed: 42);

            algorithm.Initialize(min: -0.1f, max: 0.1f);

            var populationBefore = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(populationBefore);

            var fitness = new float[algorithm.PopulationSize];
            for (var i = 0; i < fitness.Length; i++)
            {
                fitness[i] = (float)i;
            }

            algorithm.Tell(fitness);

            Assert.Equal(1, algorithm.Generation);
            Assert.Equal(255f, algorithm.BestFitness, 5);

            var populationAfter = new float[algorithm.PopulationSize * algorithm.ParameterCount];
            algorithm.Ask(populationAfter);

            for (var i = 0; i < populationAfter.Length; i++)
            {
                Assert.False(float.IsNaN(populationAfter[i]) || float.IsInfinity(populationAfter[i]),
                    $"Position {i} contains NaN or Inf.");
            }

            var expectedGenome = SliceGenome(populationBefore, 255, algorithm.ParameterCount);

            for (var genomeIndex = 0; genomeIndex < algorithm.PopulationSize; genomeIndex++)
            {
                var genome = SliceGenome(populationAfter, genomeIndex, algorithm.ParameterCount);
                Assert.Equal(expectedGenome, genome);
            }
        }

        [Fact]
        public void Tell_WithCrossover_PreservesPopulationStructure()
        {
            // With crossover enabled, the GA must still produce a full population of the
            // correct size on every generation, eliteFraction × popSize of those must be
            // unchanged elites, and Generation must still increment exactly once per Tell.
            // Uses CopyMutation + a deterministic crossover that returns parent1 as child1
            // and parent2 as child2 — so we can reason about the final population directly.
            using var algorithm = new GenerationalGeneticAlgorithm(
                populationSize: 8,
                parameterCount: 2,
                eliteFraction: 0.25f, // 2 elites
                selectionOperator: new FirstEliteSelectionOperator(),
                mutationOperator: new CopyMutationOperator(),
                fitnessShaper: null,
                seed: 123,
                crossoverOperator: new IdentityCrossoverOperator());

            algorithm.Initialize();

            var before = new float[8 * 2];
            algorithm.Ask(before);

            // With strictly increasing fitness, the top elite is at index 7, second at 6.
            float[] fitness = [0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f];

            algorithm.Tell(fitness);

            var after = new float[8 * 2];
            algorithm.Ask(after);

            Assert.Equal(1, algorithm.Generation);
            Assert.Equal(7f, algorithm.BestFitness, 5);

            // First elite slot (index 0 in the new population) should be the best from before.
            var bestGenome = SliceGenome(before, 7, 2);
            Assert.Equal(bestGenome, SliceGenome(after, 0, 2));
        }

        [Fact]
        public void Tell_WithCrossover_IsAllocationStable()
        {
            using var algorithm = new GenerationalGeneticAlgorithm(
                populationSize: 16,
                parameterCount: 4,
                eliteFraction: 0.25f,
                selectionOperator: new FirstEliteSelectionOperator(),
                mutationOperator: new CopyMutationOperator(),
                fitnessShaper: null,
                seed: 123,
                crossoverOperator: new IdentityCrossoverOperator());

            algorithm.Initialize();
            var fitness = new float[16];
            for (var i = 0; i < fitness.Length; i++)
            {
                fitness[i] = i;
            }
            algorithm.Tell(fitness); // warmup

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 5_000; i++)
            {
                algorithm.Tell(fitness);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated <= 512, $"Tell with crossover allocated {allocated} bytes.");
        }

        [Fact]
        public void Tell_WithCrossover_HandlesOddNonEliteCount()
        {
            // Non-elite count is odd: crossover produces pairs, so the final slot is filled
            // by mutation-only. The assertion just confirms the GA finishes Tell without
            // throwing or leaving holes.
            using var algorithm = new GenerationalGeneticAlgorithm(
                populationSize: 7, // 1 elite (round(7 * 0.1) = 1), 6 children — even, but
                parameterCount: 2, // we force odd by using eliteFraction = 0.3 -> 2 elites, 5 children
                eliteFraction: 0.3f,
                selectionOperator: new FirstEliteSelectionOperator(),
                mutationOperator: new CopyMutationOperator(),
                fitnessShaper: null,
                seed: 123,
                crossoverOperator: new IdentityCrossoverOperator());

            algorithm.Initialize();
            var fitness = new float[7];
            for (var i = 0; i < fitness.Length; i++)
            {
                fitness[i] = i;
            }

            algorithm.Tell(fitness);

            Assert.Equal(1, algorithm.Generation);
            Assert.Equal(6f, algorithm.BestFitness, 5);
        }

        private static GenerationalGeneticAlgorithm CreateAlgorithm(
            ISelectionOperator selection,
            IMutationOperator mutation,
            int populationSize = 4,
            int parameterCount = 3,
            float eliteFraction = 0.5f)
        {
            return new GenerationalGeneticAlgorithm(
            populationSize,
            parameterCount,
            eliteFraction,
            selection,
            mutation,
            fitnessShaper: null,
            seed: 123);
        }

        private static float[] SliceGenome(float[] population, int bestGenomeIndex, int parameterCount)
        {
            var result = new float[parameterCount];
            Array.Copy(population, bestGenomeIndex * parameterCount, result, 0, parameterCount);
            return result;
        }

        private sealed class FirstEliteSelectionOperator : ISelectionOperator
        {
            public int SelectParent(ReadOnlySpan<int> eliteIndices, Random rng) => eliteIndices[0];
        }

        private sealed class CopyMutationOperator : IMutationOperator
        {
            public void Mutate(ReadOnlySpan<float> parentGenome, Span<float> childGenome, Random rng)
            {
                parentGenome.CopyTo(childGenome);
            }
        }

        private sealed class AddOneMutationOperator : IMutationOperator
        {
            public void Mutate(ReadOnlySpan<float> parentGenome, Span<float> childGenome, Random rng)
            {
                parentGenome.CopyTo(childGenome);

                for (var i = 0; i < childGenome.Length; i++)
                {
                    childGenome[i] += 1f;
                }
            }
        }

        /// <summary>
        ///     Crossover that copies parent1 to child1 and parent2 to child2 verbatim.
        ///     Removes crossover randomness from integration tests so the remaining GA
        ///     machinery (elite copying, step=2 iteration, odd-leftover fallback) can be
        ///     verified deterministically.
        /// </summary>
        private sealed class IdentityCrossoverOperator : ICrossoverOperator
        {
            public void Crossover(
                ReadOnlySpan<float> parent1,
                ReadOnlySpan<float> parent2,
                Span<float> child1,
                Span<float> child2,
                Random rng)
            {
                parent1.CopyTo(child1);
                parent2.CopyTo(child2);
            }
        }
    }
}