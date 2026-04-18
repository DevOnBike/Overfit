using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Evolutionary.Storage
{
    public sealed class EvolutionWorkspace : IDisposable
    {
        private int _disposed;

        public EvolutionWorkspace(int populationSize, int genomeSize, bool clearMemory = false)
        {
            PopulationSize = populationSize;
            GenomeSize = genomeSize;

            Population = new FastTensor<float>(populationSize, genomeSize, clearMemory);
            NextPopulation = new FastTensor<float>(populationSize, genomeSize, clearMemory);
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

        public FastTensor<float> Population { get; }
        public FastTensor<float> NextPopulation { get; }
        public FastTensor<float> Fitness { get; }
        public FastTensor<float> ShapedFitness { get; }

        public int[] Ranking { get; }
        public int[] EliteIndices { get; }

        public void SwapPopulations()
        {
            var current = Population.GetView().AsSpan();
            var next = NextPopulation.GetView().AsSpan();
            next.CopyTo(current);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            Population.Dispose();
            NextPopulation.Dispose();
            Fitness.Dispose();
            ShapedFitness.Dispose();
        }
    }
}

