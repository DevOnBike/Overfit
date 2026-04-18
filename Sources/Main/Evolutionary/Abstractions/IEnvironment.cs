namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Reprezentuje środowisko (lub symulację), które ocenia populację.
    /// W Twoim przypadku to może być symulacja roju lub klient w Unity.
    /// </summary>
    public interface IEnvironment
    {
        /// <summary>
        /// Otrzymuje tensor populacji [PopulationSize, GenomeSize] 
        /// i zwraca tensor/Span z wynikami fitness [PopulationSize].
        /// </summary>
        void Evaluate(ReadOnlySpan<float> populationData, Span<float> fitnessOut, int populationSize, int genomeSize);
    }

}