using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Selection
{
    public sealed class TruncationSelectionOperator : ISelectionOperator
    {
        public int SelectParent(ReadOnlySpan<int> eliteIndices, Random rng)
        {
            if (eliteIndices.Length == 0)
            {
                throw new ArgumentException("Elite set nie może być pusty.", nameof(eliteIndices));
            }

            return eliteIndices[rng.Next(eliteIndices.Length)];
        }
        public void Select(ReadOnlySpan<float> fitness, Span<int> selectedIndicesOut)
        {
            throw new NotImplementedException();
        }
    }
}