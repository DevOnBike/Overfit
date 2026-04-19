namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IEvolutionAlgorithm : IDisposable, IEvolutionCheckpoint
    {
        int PopulationSize { get; }
        int ParameterCount { get; }
        int Generation { get; }

        void Ask(Span<float> populationMatrix);   // [populationSize * parameterCount]
        void Tell(ReadOnlySpan<float> fitness);

        ReadOnlySpan<float> GetBestParameters();

        float BestFitness { get; }
    }
}