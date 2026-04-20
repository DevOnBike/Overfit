using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Selection
{
    /// <summary>
    ///     Obsolete alias preserved for source-compatibility with pre-rename callers.
    ///     Prefer <see cref="UniformEliteParentSelector"/> � the behavior is identical and
    ///     the name is more accurate (this operator does uniform selection over a pre-truncated
    ///     elite set; the truncation itself happens in the enclosing algorithm).
    /// </summary>
    [Obsolete("Renamed to UniformEliteParentSelector to better reflect its behavior. The name 'TruncationSelectionOperator' was misleading because the truncation step happens earlier in GenerationalGeneticAlgorithm.CreateNextGeneration, not in this selector.")]
    public sealed class TruncationSelectionOperator : ISelectionOperator
    {
        private readonly UniformEliteParentSelector _inner = new();

        public int SelectParent(ReadOnlySpan<int> eliteIndices, Random rng)
        {
            return _inner.SelectParent(eliteIndices, rng);
        }
    }
}