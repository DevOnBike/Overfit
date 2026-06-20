// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// Result of a paraphrase-stability evaluation: per-group mean pairwise Jaccard overlap of the retrieved
    /// id sets, and which groups fell below the stability bar. A low score means rephrasing a question swaps the
    /// retrieved sources — a brittle retriever.
    /// </summary>
    public sealed class ParaphraseStabilityReport
    {
        public ParaphraseStabilityReport(int topK, double minJaccard, IReadOnlyList<GroupResult> groups)
        {
            ArgumentNullException.ThrowIfNull(groups);
            TopK = topK;
            MinJaccard = minJaccard;
            Groups = groups;

            var sum = 0.0;
            var unstable = 0;
            for (var i = 0; i < groups.Count; i++)
            {
                sum += groups[i].MeanJaccard;
                if (!groups[i].IsStable)
                {
                    unstable++;
                }
            }

            MeanStability = groups.Count == 0 ? 0.0 : sum / groups.Count;
            UnstableCount = unstable;
        }

        public int TopK
        {
            get;
        }

        /// <summary>The minimum mean Jaccard for a group to count as stable.</summary>
        public double MinJaccard
        {
            get;
        }

        /// <summary>Mean stability across all groups.</summary>
        public double MeanStability
        {
            get;
        }

        /// <summary>How many groups fell below <see cref="MinJaccard"/>.</summary>
        public int UnstableCount
        {
            get;
        }

        public IReadOnlyList<GroupResult> Groups
        {
            get;
        }

        /// <summary>One paraphrase group's stability: the mean pairwise overlap and the per-variant retrieved ids.</summary>
        public sealed class GroupResult
        {
            public GroupResult(string name, double meanJaccard, bool isStable, IReadOnlyList<IReadOnlyList<string>> retrievedPerVariant)
            {
                Name = name;
                MeanJaccard = meanJaccard;
                IsStable = isStable;
                RetrievedPerVariant = retrievedPerVariant;
            }

            public string Name
            {
                get;
            }

            /// <summary>Mean pairwise Jaccard overlap of the retrieved id sets across the group's variants (0..1).</summary>
            public double MeanJaccard
            {
                get;
            }

            public bool IsStable
            {
                get;
            }

            /// <summary>The retrieved top-K id list for each variant, in variant order.</summary>
            public IReadOnlyList<IReadOnlyList<string>> RetrievedPerVariant
            {
                get;
            }
        }
    }
}
