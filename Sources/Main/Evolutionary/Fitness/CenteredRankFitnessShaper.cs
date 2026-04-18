using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Fitness
{
    public sealed class CenteredRankFitnessShaper : IFitnessShaper
    {
        public void Shape(ReadOnlySpan<float> rawFitness, Span<float> shapedFitness)
        {
            if (rawFitness.Length != shapedFitness.Length)
            {
                throw new ArgumentException("rawFitness i shapedFitness muszą mieć ten sam rozmiar.");
            }

            var count = rawFitness.Length;
            if (count == 0)
            {
                return;
            }

            var pairs = new (float fitness, int index)[count];
            for (var i = 0; i < count; i++)
            {
                pairs[i] = (rawFitness[i], i);
            }

            Array.Sort(pairs, static (a, b) => a.fitness.CompareTo(b.fitness));

            if (count == 1)
            {
                shapedFitness[pairs[0].index] = 0f;
                return;
            }

            for (var rank = 0; rank < count; rank++)
            {
                var normalized = (float)rank / (count - 1);
                shapedFitness[pairs[rank].index] = normalized - 0.5f;
            }
        }
    }
}