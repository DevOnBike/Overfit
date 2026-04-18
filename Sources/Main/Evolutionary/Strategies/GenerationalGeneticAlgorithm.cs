using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Strategies
{
    public sealed class GenerationalGeneticAlgorithm : IEvolutionAlgorithm
    {

        public void Dispose()
        {
            throw new NotImplementedException();
        }
        public int PopulationSize { get; }
        public int ParameterCount { get; }
        public int Generation { get; }
        public void Ask(Span<float> populationMatrix)
        {
            throw new NotImplementedException();
        }
        public void Tell(ReadOnlySpan<float> fitness)
        {
            throw new NotImplementedException();
        }
        public ReadOnlySpan<float> GetBestParameters()
        {
            throw new NotImplementedException();
        }
        public float BestFitness { get; }
    }
}