using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Evolutionary.Storage
{
    public sealed class EvolutionWorkspace : IDisposable
    {
        private int _disposed;
        private FastTensor<float> _population;
        private FastTensor<float> _nextPopulation;

        public EvolutionWorkspace(int populationSize, int genomeSize, bool clearMemory = false)
        {
            if (populationSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(populationSize));
            }

            if (genomeSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(genomeSize));
            }

            PopulationSize = populationSize;
            GenomeSize = genomeSize;

            _population = new FastTensor<float>(populationSize, genomeSize, clearMemory);
            _nextPopulation = new FastTensor<float>(populationSize, genomeSize, clearMemory);
            Fitness = new FastTensor<float>(populationSize, clearMemory);
            ShapedFitness = new FastTensor<float>(populationSize, clearMemory);

            Ranking = new int[populationSize];
            EliteIndices = new int[populationSize];

            for (var i = 0; i < populationSize; i++)
            {
                Ranking[i] = i;
                EliteIndices[i] = i;
            }
        }

        public int PopulationSize { get; }

        public int GenomeSize { get; }

        public FastTensor<float> Population => _population;

        public FastTensor<float> NextPopulation => _nextPopulation;

        public FastTensor<float> Fitness { get; }

        public FastTensor<float> ShapedFitness { get; }

        public int[] Ranking { get; }

        public int[] EliteIndices { get; }

        public void SwapPopulations()
        {
            ThrowIfDisposed();
            (_population, _nextPopulation) = (_nextPopulation, _population);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            _population.Dispose();
            _nextPopulation.Dispose();
            Fitness.Dispose();
            ShapedFitness.Dispose();
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);
        }
    }
}
