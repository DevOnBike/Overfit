// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Selection
{
    /// <summary>
    ///     Tournament selection over the already-sorted elite set. Draws
    ///     <see cref="TournamentSize"/> positions uniformly at random from the elite array
    ///     and returns the genome index at the smallest drawn position — which, since the
    ///     elite set is sorted from best to worst, is the fittest of the drawn candidates.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Compared with <c>TruncationSelectionOperator</c> (uniform pick over elites),
    ///         tournament selection provides stronger selective pressure without the
    ///         premature convergence that fitness-proportional (roulette) selection can
    ///         cause. Higher <see cref="TournamentSize"/> means stronger pressure:
    ///         <list type="bullet">
    ///             <item>k = 1 reduces to uniform random over elites (same as truncation).</item>
    ///             <item>k = 3 is the textbook default.</item>
    ///             <item>k = |elites| picks the best elite every time.</item>
    ///         </list>
    ///     </para>
    ///     <para>
    ///         This implementation operates on the elite set rather than the entire
    ///         population, matching the existing <see cref="ISelectionOperator"/> contract.
    ///         That is equivalent to classic tournament-on-population once the population is
    ///         partitioned into elites + non-elites — only the elites survive to breed
    ///         anyway in the GA's generational replacement scheme.
    ///     </para>
    ///     <para>
    ///         Zero-allocation. The implementation tracks the smallest-position draw in a
    ///         single int, so the tournament fits entirely in registers for any size.
    ///     </para>
    /// </remarks>
    public sealed class TournamentSelectionOperator : ISelectionOperator
    {
        public TournamentSelectionOperator(int tournamentSize = 3)
        {
            if (tournamentSize < 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tournamentSize),
                    "Tournament size must be at least 1.");
            }

            TournamentSize = tournamentSize;
        }

        public int TournamentSize { get; }

        public int SelectParent(ReadOnlySpan<int> eliteIndices, Random rng)
        {
            if (eliteIndices.Length == 0)
            {
                throw new ArgumentException("Elite set cannot be empty.", nameof(eliteIndices));
            }

            // Fast scalar path: track the smallest drawn position directly, no storage.
            // Because eliteIndices is pre-sorted descending by fitness, the smallest drawn
            // position points at the highest-fitness candidate among the draws.
            var best = int.MaxValue;
            var k = TournamentSize;
            var rangeExclusive = eliteIndices.Length;

            for (var i = 0; i < k; i++)
            {
                var drawn = rng.Next(rangeExclusive);

                if (drawn < best)
                {
                    best = drawn;
                }
            }

            return eliteIndices[best];
        }
    }
}