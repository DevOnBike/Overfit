namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Główny orkiestrator algorytmu.
    /// </summary>
    public interface IEvolutionStrategy
    {
        void Step(IEnvironment environment);
    }
}