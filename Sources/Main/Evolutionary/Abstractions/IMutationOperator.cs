namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Mutates a parent genome into a child genome without allocating.
    /// </summary>
    public interface IMutationOperator
    {
        void Mutate(ReadOnlySpan<float> parentGenome, Span<float> childGenome, Random rng);
    }
}