// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;
using DevOnBike.Overfit.Maths;

namespace DevOnBike.Overfit.Evolutionary.Fitness
{
    /// <summary>
    ///     Maps raw fitness values onto a centered rank in [-0.5, +0.5]:
    ///     the worst individual receives -0.5, the best +0.5, with ranks linearly
    ///     interpolated in between. Ties are broken by index; NaN values rank as worst.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Standard fitness shaping for noisy black-box optimizers (Salimans et al., 2017).
    ///         The transform is invariant to monotone rescalings of the fitness function and
    ///         prevents a single outlier individual from dominating the search direction.
    ///     </para>
    ///     <para>
    ///         Instances carry a pooled ranking buffer that grows monotonically with the largest
    ///         population seen so far, so steady-state calls perform zero managed allocations.
    ///         Not thread-safe: use one shaper per evolutionary algorithm instance.
    ///     </para>
    /// </remarks>
    public sealed class CenteredRankFitnessShaper : IFitnessShaper
    {
        private int[] _ranking = [];

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

            if (_ranking.Length < count)
            {
                _ranking = new int[count];
            }

            var ranking = _ranking;

            for (var i = 0; i < count; i++)
            {
                ranking[i] = i;
            }

            // Ascending: worst fitness first (rank 0 -> -0.5), best last (rank n-1 -> +0.5).
            PartialSort.SortIndices(ranking, rawFitness, ascending: true);

            var denominator = (float)(count - 1);

            for (var rank = 0; rank < count; rank++)
            {
                var normalized = rank / denominator;
                shapedFitness[ranking[rank]] = normalized - 0.5f;
            }
        }
    }
}