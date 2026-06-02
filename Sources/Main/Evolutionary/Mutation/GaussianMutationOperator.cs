// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Randomization;

namespace DevOnBike.Overfit.Evolutionary.Mutation
{
    /// <summary>
    ///     Per-element Gaussian mutation: each gene independently has probability
    ///     <c>mutationProbability</c> of receiving an additive N(0, sigma^2)
    ///     perturbation, with the result clamped to
    ///     [<c>minWeight</c>, <c>maxWeight</c>].
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Standard-normal samples are drawn via the Box-Muller transform with paired-sample
    ///         caching: every pair of uniform draws produces two independent standard-normal
    ///         samples, and the unused one is retained and returned on the next draw. This
    ///         halves the number of <see cref="MathF.Sqrt"/>, <c>MathF.Log</c>, and
    ///         <see cref="MathF.SinCos"/> operations compared with the textbook Box-Muller that
    ///         discards half of each pair. The spare lives in a local for the duration of a
    ///         single <see cref="Mutate"/> call and is never carried across calls, so a
    ///         genome's perturbation depends only on the supplied <see cref="IRandom"/> — which
    ///         is what lets a run be replayed bit-identically from the RNG state alone.
    ///     </para>
    ///     <para>
    ///         The operator holds no mutable instance or static state, so it is safe to invoke
    ///         concurrently from multiple threads as long as each call is given its own
    ///         <see cref="IRandom"/>.
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

            ArgumentOutOfRangeException.ThrowIfLessThan(sigma, 0f);

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
            IRandom rng)
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

            // Box-Muller spare, scoped to this call. Resetting it per call keeps the
            // genome's randomness a pure function of rng's state on entry.
            var spareGaussian = 0f;
            var hasSpareGaussian = false;

            for (var i = 0; i < childGenome.Length; i++)
            {
                // NextSingle returns a float in [0, 1). Cheaper than NextDouble; Box-Muller below
                // stays in the float domain end-to-end.
                if (rng.NextSingle() >= probability)
                {
                    continue;
                }

                var gaussian = NextGaussian(rng, ref spareGaussian, ref hasSpareGaussian);
                var perturbed = childGenome[i] + (gaussian * sigma);
                childGenome[i] = Math.Clamp(perturbed, minWeight, maxWeight);
            }
        }

        private static float NextGaussian(IRandom rng, ref float spareGaussian, ref bool hasSpareGaussian)
        {
            if (hasSpareGaussian)
            {
                hasSpareGaussian = false;
                return spareGaussian;
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

            spareGaussian = mag * sin;
            hasSpareGaussian = true;

            return mag * cos;
        }
    }
}