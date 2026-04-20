namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    ///     Combines two parent genomes into two child genomes without allocating.
    ///     Implementations must produce both children in a single call — this matches the
    ///     natural pairing of crossover operators from the genetic-algorithm literature
    ///     (SBX, BLX-α, arithmetic, etc.), which generate offspring in pairs.
    /// </summary>
    /// <remarks>
    ///     Implementations must accept parent spans that alias child spans only when the
    ///     caller explicitly documents it; the safe assumption is that all four spans are
    ///     disjoint. All operators receive an external <see cref="Random"/> so the caller
    ///     controls determinism and thread-local RNG state.
    /// </remarks>
    public interface ICrossoverOperator
    {
        void Crossover(ReadOnlySpan<float> parent1, ReadOnlySpan<float> parent2, Span<float> child1, Span<float> child2, Random rng);
    }
}