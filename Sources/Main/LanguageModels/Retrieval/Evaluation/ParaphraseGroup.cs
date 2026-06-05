// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// A set of <see cref="Variants"/> that all ask the SAME thing in different words. A stable RAG retrieves
    /// (nearly) the same top-K documents for every variant — if rephrasing a question swaps the retrieved
    /// sources, the system is brittle. Measured as the mean pairwise Jaccard overlap of the retrieved id sets.
    /// </summary>
    public sealed class ParaphraseGroup
    {
        public ParaphraseGroup(string name, params string[] variants)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(name);
            ArgumentNullException.ThrowIfNull(variants);
            if (variants.Length < 2)
            {
                throw new ArgumentException("A paraphrase group needs at least two variants to compare.", nameof(variants));
            }

            Name = name;
            Variants = variants;
        }

        /// <summary>A label for the intent this group expresses (for reporting).</summary>
        public string Name { get; }

        /// <summary>Two or more rephrasings of the same question.</summary>
        public IReadOnlyList<string> Variants { get; }
    }
}
