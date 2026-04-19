using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Evolutionary.Fitness;
using DevOnBike.Overfit.Evolutionary.Mutation;
using DevOnBike.Overfit.Evolutionary.Selection;
using DevOnBike.Overfit.Evolutionary.Strategies;

namespace Benchmarks
{
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class GenerationalGeneticAlgorithmBenchmarks
    {
        private GenerationalGeneticAlgorithm _algorithm = null!;
        private float[] _population = null!;
        private float[] _fitness = null!;
        private Random _fitnessRng = null!;

        [Params(256, 1024)]
        public int PopulationSize { get; set; }

        [Params(64, 256)]
        public int ParameterCount { get; set; }

        /// <summary>
        ///     Selects the combination of operators under test:
        ///     Minimal = truncation selection + copy mutation + no shaper (pure GA skeleton cost).
        ///     Realistic = truncation selection + Gaussian mutation + centered-rank shaper
        ///     (what production pipelines actually run per generation).
        /// </summary>
        [Params(Config.Minimal, Config.Realistic)]
        public Config Configuration { get; set; }

        public enum Config
        {
            Minimal,
            Realistic,
        }

        [GlobalSetup]
        public void Setup()
        {
            ISelectionOperator selection;
            IMutationOperator mutation;
            IFitnessShaper? shaper;

            switch (Configuration)
            {
                case Config.Realistic:
                    selection = new TruncationSelectionOperator();
                    mutation = new GaussianMutationOperator();
                    shaper = new CenteredRankFitnessShaper();
                    break;
                case Config.Minimal:
                default:
                    selection = new FirstEliteSelectionOperator();
                    mutation = new CopyMutationOperator();
                    shaper = null;
                    break;
            }

            _algorithm = new GenerationalGeneticAlgorithm(
                PopulationSize,
                ParameterCount,
                eliteFraction: 0.1f,
                selectionOperator: selection,
                mutationOperator: mutation,
                fitnessShaper: shaper,
                seed: 123);

            _algorithm.Initialize(min: -0.25f, max: 0.25f);

            _population = new float[PopulationSize * ParameterCount];
            _fitness = new float[PopulationSize];
            _fitnessRng = new Random(7919);

            RandomizeFitness();

            _algorithm.Ask(_population);
            _algorithm.Tell(_fitness);
        }

        // Reshuffle fitness between iterations so the sorter never sees the same input twice.
        // Without this, JIT + branch predictor + insertion-sort best-case conspire to make
        // the "Tell" benchmark run essentially as O(n) regardless of the sort algorithm.
        [IterationSetup(Target = nameof(Tell))]
        public void ShuffleFitnessBeforeTell()
        {
            RandomizeFitness();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _algorithm.Dispose();
        }

        [Benchmark]
        public void Ask()
        {
            _algorithm.Ask(_population);
        }

        [Benchmark]
        public void Tell()
        {
            _algorithm.Tell(_fitness);
        }

        private void RandomizeFitness()
        {
            for (var i = 0; i < _fitness.Length; i++)
            {
                _fitness[i] = _fitnessRng.NextSingle() * 1000f;
            }
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
    }
}