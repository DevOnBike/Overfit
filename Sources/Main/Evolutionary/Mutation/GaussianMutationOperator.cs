using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Mutation
{
    public sealed class GaussianMutationOperator : IMutationOperator
    {
        private readonly float _mutationProbability;
        private readonly float _sigma;
        private readonly float _minWeight;
        private readonly float _maxWeight;

        public GaussianMutationOperator(
            float mutationProbability = 0.08f,
            float sigma = 0.05f,
            float minWeight = -2.5f,
            float maxWeight = 2.5f)
        {
            if (mutationProbability is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(mutationProbability));
            }

            if (sigma < 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(sigma));
            }

            if (minWeight > maxWeight)
            {
                throw new ArgumentException("minWeight nie może być większy niż maxWeight.");
            }

            _mutationProbability = mutationProbability;
            _sigma = sigma;
            _minWeight = minWeight;
            _maxWeight = maxWeight;
        }
        
        public void Mutate(Span<float> populationData, int populationSize, int genomeSize)
        {
            throw new NotImplementedException();
        }

        public void Mutate(ReadOnlySpan<float> parentGenome, Span<float> childGenome, Random rng)
        {
            if (parentGenome.Length != childGenome.Length)
            {
                throw new ArgumentException("parentGenome i childGenome muszą mieć ten sam rozmiar.");
            }

            parentGenome.CopyTo(childGenome);

            for (var i = 0; i < childGenome.Length; i++)
            {
                if (rng.NextDouble() > _mutationProbability)
                {
                    continue;
                }

                childGenome[i] = Math.Clamp(childGenome[i] + NextGaussian(rng) * _sigma, _minWeight, _maxWeight);
            }
        }

        private static float NextGaussian(Random rng)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            var stdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return (float)stdNormal;
        }
        
        
    }
}