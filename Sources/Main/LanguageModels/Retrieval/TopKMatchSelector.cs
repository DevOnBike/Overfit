// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// Shared top-K insertion for retrieval result spans: keeps the best <c>k</c> candidates in descending
    /// score order without sorting the corpus and without allocating. Candidates arrive in arbitrary order
    /// and are shifted into place, so a full scan costs O(n·k) worst case but O(n) once the list is warm and
    /// most candidates are rejected by the first comparison.
    ///
    /// <para><see cref="VectorStore"/> keeps its own private copy of this routine — it predates the shared
    /// helper and is covered by its own tests, so it was deliberately left untouched rather than refactored
    /// for tidiness alone.</para>
    /// </summary>
    internal static class TopKMatchSelector
    {
        /// <summary>
        /// Offers <paramref name="candidate"/> to the descending top-<paramref name="k"/> list held in
        /// <paramref name="results"/>, updating <paramref name="found"/> (the number of populated slots).
        /// </summary>
        internal static void InsertDescending(
            Span<VectorMatch> results,
            ref int found,
            int k,
            in VectorMatch candidate)
        {
            // Reject early when the list is full and the candidate cannot beat the current worst.
            if (found == k && candidate.Score <= results[k - 1].Score)
            {
                return;
            }

            var position = found < k ? found : k - 1;
            while (position > 0 && results[position - 1].Score < candidate.Score)
            {
                results[position] = results[position - 1];
                position--;
            }

            results[position] = candidate;

            if (found < k)
            {
                found++;
            }
        }
    }
}
