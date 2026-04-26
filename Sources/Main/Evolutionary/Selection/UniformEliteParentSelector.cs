// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Selection
{
    /// <summary>
    ///     Picks a parent uniformly at random from the already-sorted elite set.
    ///     This is the simplest legal <see cref="ISelectionOperator"/>: every elite has equal
    ///     probability of being chosen, independent of its rank within the elite set.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Naming note: this operator is sometimes called "truncation selection" in textbooks,
    ///         but the actual <em>truncation</em> — i.e. the decision about which candidates become
    ///         elites — happens earlier in <c>GenerationalGeneticAlgorithm</c> via
    ///         <c>PartialSort.TopKDescending</c>. By the time this operator is invoked the elite
    ///         set is a fixed input; its job is parent <em>selection</em>, not truncation.
    ///         The name was changed to reflect that.
    ///     </para>
    ///     <para>
    ///         For stronger selective pressure without switching to fitness-proportional (roulette)
    ///         selection, use <see cref="TournamentSelectionOperator"/> instead.
    ///     </para>
    /// </remarks>
    public sealed class UniformEliteParentSelector : ISelectionOperator
    {
        public int SelectParent(ReadOnlySpan<int> eliteIndices, Random rng)
        {
            if (eliteIndices.Length == 0)
            {
                throw new ArgumentException("Elite set cannot be empty.", nameof(eliteIndices));
            }

            return eliteIndices[rng.Next(eliteIndices.Length)];
        }
    }

    
}
