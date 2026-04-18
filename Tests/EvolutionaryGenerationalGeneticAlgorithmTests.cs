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

            algorithm.Tell([4f, 3f, 2f, 1f]); // warmup

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 5_000; i++)
            {
                algorithm.Tell([4f, 3f, 2f, 1f]);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.True(allocated <= 512, $"Tell allocated {allocated} bytes.");
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
    }
}