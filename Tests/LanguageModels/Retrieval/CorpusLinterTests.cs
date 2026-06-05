// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;
using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// The corpus linter flags documents that quietly hurt retrieval — near-duplicates (redundant passages) and
    /// orphans (entries no query can reach). Deterministic over a tiny crafted corpus, no model required.
    /// </summary>
    public sealed class CorpusLinterTests
    {
        private const int Dim = 5;

        private static VectorStore BuildStore()
        {
            var store = new VectorStore(Dim);
            store.Add("paris", [1f, 0f, 0f, 0f, 0f], "Paris is the capital of France.");
            store.Add("paris-dup", [0.999f, 0.01f, 0f, 0f, 0f], "Paris, France's capital city.");  // near-duplicate of paris
            store.Add("berlin", [0f, 1f, 0f, 0f, 0f], "Berlin is the capital of Germany.");
            store.Add("madrid", [0f, 0f, 0f, 1f, 0f], "Madrid is the capital of Spain.");           // no query reaches it
            return store;
        }

        [Fact]
        public void FindNearDuplicates_FlagsTheRedundantPair()
        {
            var linter = new CorpusLinter(BuildStore());

            var dups = linter.FindNearDuplicates(threshold: 0.97);

            Assert.Single(dups);
            Assert.Equal("paris", dups[0].FirstId);          // canonical (ordinal) ordering
            Assert.Equal("paris-dup", dups[0].SecondId);
            Assert.True(dups[0].Similarity > 0.97f, $"similarity was {dups[0].Similarity}");
        }

        [Fact]
        public void FindNearDuplicates_None_WhenAllDistinct()
        {
            var store = new VectorStore(Dim);
            store.Add("a", [1f, 0f, 0f, 0f, 0f]);
            store.Add("b", [0f, 1f, 0f, 0f, 0f]);
            store.Add("c", [0f, 0f, 1f, 0f, 0f]);

            var dups = new CorpusLinter(store).FindNearDuplicates(threshold: 0.97);

            Assert.Empty(dups);
        }

        [Fact]
        public void FindOrphans_ReturnsUnreachableDocuments()
        {
            var linter = new CorpusLinter(BuildStore());

            // Queries that reach paris / paris-dup / berlin, but never the madrid axis.
            var queries = new float[][]
            {
                [1f, 0f, 0f, 0f, 0f],
                [0f, 1f, 0f, 0f, 0f],
            };

            var orphans = linter.FindOrphans(queries, topK: 2);

            Assert.Single(orphans);
            Assert.Equal("madrid", orphans[0]);
        }
    }
}
