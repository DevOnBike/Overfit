// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Mutation
{
    /// <summary>
    ///     Per-element Gaussian mutation: each gene independently has probability
    ///     <paramref name="mutationProbability"/> of receiving an additive
    ///     N(0, sigma^2) perturbation, with the result clamped to
    ///     [<paramref name="minWeight"/>, <paramref name="maxWeight"/>].
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Standard-normal samples are drawn via the Box-Muller transform with paired-sample
    ///         caching: every pair of uniform draws produces two independent standard-normal
    ///         samples, and the unused one is retained in a thread-local field and returned on
    ///         the next call. This halves the number of <see cref="MathF.Sqrt"/>,
    ///         <see cref="MathF.Log"/>, and <see cref="MathF.SinCos"/> operations compared with
    ///         the textbook Box-Muller that discards half of each pair.
    ///     </para>
    ///     <para>
    ///         The spare-sample cache is <see cref="ThreadStaticAttribute">[ThreadStatic]</see>,
    ///         which makes the operator safe to invoke concurrently from multiple threads —
    ///         each thread maintains its own half-pair independently. There is no cross-thread
    ///         synchronization; the operator remains stateless at the instance level.
    ///     </para>
    ///     <para>
    ///         Runs entirely in <c>float</c> via <see cref="MathF"/> intrinsics rather than
    ///         promoting to <see cref="double"/>, which avoids two implicit conversions per sample
    ///         and keeps register pressure predictable for the JIT. <see cref="MathF.SinCos"/>
    ///         (available since .NET 7) computes both trig values from a single argument reduction.
    ///     </para>
    /// </remarks>
    public sealed class GaussianMutationOperator : IMutationOperator
    {
        // Thread-local Box-Muller spare. The transform produces two independent standard-normal
        // samples per pair of uniform draws; we return one immediately and stash the other here
        // for the next call on the same thread. Indexed by thread via [ThreadStatic].
        [ThreadStatic]
        private static float _spareGaussian;

        [ThreadStatic]
        private static bool _hasSpareGaussian;

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
                throw new ArgumentException("minWeight cannot be greater than maxWeight.");
            }

            _mutationProbability = mutationProbability;
            _sigma = sigma;
            _minWeight = minWeight;
            _maxWeight = maxWeight;
        }

        public void Mutate(
            ReadOnlySpan<float> parentGenome,
            Span<float> childGenome,
            Random rng)
        {
            if (parentGenome.Length != childGenome.Length)
            {
                throw new ArgumentException("parentGenome and childGenome must have the same length.");
            }

            parentGenome.CopyTo(childGenome);

            // Cache mutable state in locals so the JIT can keep them in registers inside the hot loop.
            var probability = _mutationProbability;
            var sigma = _sigma;
            var minWeight = _minWeight;
            var maxWeight = _maxWeight;

            for (var i = 0; i < childGenome.Length; i++)
            {
                // NextSingle returns a float in [0, 1). Cheaper than NextDouble; Box-Muller below
                // stays in the float domain end-to-end.
                if (rng.NextSingle() >= probability)
                {
                    continue;
                }

                var perturbed = childGenome[i] + NextGaussian(rng) * sigma;
                childGenome[i] = Math.Clamp(perturbed, minWeight, maxWeight);
            }
        }

        private static float NextGaussian(Random rng)
        {
            if (_hasSpareGaussian)
            {
                _hasSpareGaussian = false;
                return _spareGaussian;
            }

            // Textbook Box-Muller. Reject u1 == 0 to keep Log finite. The probability of drawing
            // exactly zero from NextSingle is ~2^-24, so the loop runs effectively once.
            float u1;

            do
            {
                u1 = rng.NextSingle();
            }
            while (u1 == 0f);

            var u2 = rng.NextSingle();

            var mag = MathF.Sqrt(-2f * MathF.Log(u1));
            var angle = 2f * MathF.PI * u2;

            // MathF.SinCos (.NET 7+) computes both values from a single argument reduction.
            var (sin, cos) = MathF.SinCos(angle);

            _spareGaussian = mag * sin;
            _hasSpareGaussian = true;

            return mag * cos;
        }
    }
}