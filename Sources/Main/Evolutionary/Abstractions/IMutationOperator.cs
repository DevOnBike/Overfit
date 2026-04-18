namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Operatory mutacji operujące bezpośrednio na pamięci (SIMD/No-Alloc).
    /// </summary>
    public interface IMutationOperator
    {
        void Mutate(Span<float> populationData, int populationSize, int genomeSize);
    }
}