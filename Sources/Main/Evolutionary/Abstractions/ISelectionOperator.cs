namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Selects a parent from the elite set.
    /// </summary>
    public interface ISelectionOperator
    {
        int SelectParent(ReadOnlySpan<int> eliteIndices, Random rng);
    }
}
