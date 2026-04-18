namespace DevOnBike.Overfit.Evolutionary.Fitness
{
    using DevOnBike.Overfit.Evolutionary.Abstractions;

    public sealed class CenteredRankFitnessShaper : IFitnessShaper
    {
        public void Shape(ReadOnlySpan<float> rawFitness, Span<float> shapedFitness)
        {
            if (rawFitness.Length != shapedFitness.Length)
            {
                throw new ArgumentException("rawFitness and shapedFitness must have the same length.");
            }

            var count = rawFitness.Length;
            if (count == 0)
            {
                return;
            }

            if (count == 1)
            {
                shapedFitness[0] = 0f;
                return;
            }

            var ranking = new int[count];
            for (var i = 0; i < count; i++)
            {
                ranking[i] = i;
            }

            SortIndicesAscending(ranking, rawFitness);

            for (var rank = 0; rank < count; rank++)
            {
                var normalized = (float)rank / (count - 1);
                shapedFitness[ranking[rank]] = normalized - 0.5f;
            }
        }

        private static void SortIndicesAscending(int[] indices, ReadOnlySpan<float> values)
        {
            for (var i = 1; i < indices.Length; i++)
            {
                var key = indices[i];
                var keyValue = values[key];
                var j = i - 1;

                while (j >= 0 && values[indices[j]] > keyValue)
                {
                    indices[j + 1] = indices[j];
                    j--;
                }

                indices[j + 1] = key;
            }
        }
    }
}
