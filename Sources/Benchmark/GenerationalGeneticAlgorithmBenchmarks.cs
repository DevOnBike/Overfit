using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Evolutionary.Abstractions;
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

        [Params(256, 1024)]
        public int PopulationSize { get; set; }

        [Params(64, 256)]
        public int ParameterCount { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            _algorithm = new GenerationalGeneticAlgorithm(
            PopulationSize,
            ParameterCount,
            eliteFraction: 0.1f,
            selectionOperator: new FirstEliteSelectionOperator(),
            mutationOperator: new CopyMutationOperator(),
            fitnessShaper: null,
            seed: 123);

            _algorithm.Initialize(min: -0.25f, max: 0.25f);

            _population = new float[PopulationSize * ParameterCount];
            _fitness = new float[PopulationSize];

            for (var i = 0; i < _fitness.Length; i++)
            {
                _fitness[i] = _fitness.Length - i;
            }

            _algorithm.Ask(_population);
            _algorithm.Tell(_fitness);
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
