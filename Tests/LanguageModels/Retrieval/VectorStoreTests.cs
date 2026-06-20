// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// Unit tests for the in-process vector store: cosine ranking, top-K ordering, true-cosine
    /// scores (normalisation-invariant), dimension guards, and capacity growth.
    /// </summary>
    public sealed class VectorStoreTests
    {
        [Fact]
        public void Search_RanksByCosine_BestFirst()
        {
            var store = new VectorStore(dimension: 3);
            store.Add("x", [1f, 0f, 0f], "x-axis");
            store.Add("y", [0f, 1f, 0f], "y-axis");
            store.Add("xy", [1f, 1f, 0f], "diagonal");

            var hits = store.Search([2f, 0f, 0f], topK: 3);   // points along +x (un-normalised)

            Assert.Equal(3, hits.Length);
            Assert.Equal("x", hits[0].Id);                    // cos = 1
            Assert.Equal("xy", hits[1].Id);                   // cos ≈ 0.707
            Assert.Equal("y", hits[2].Id);                    // cos = 0
            Assert.True(hits[0].Score > hits[1].Score && hits[1].Score > hits[2].Score);
        }

        [Fact]
        public void Search_ReportsTrueCosine_RegardlessOfMagnitude()
        {
            var store = new VectorStore(dimension: 2);
            store.Add("a", [3f, 0f]);                          // stored unit-normalised

            // Query points the same way but with a different magnitude → cosine is still 1.
            var hits = store.Search([10f, 0f], topK: 1);
            Assert.Single(hits);
            Assert.Equal(1f, hits[0].Score, 5);

            // Orthogonal query → cosine 0.
            Assert.Equal(0f, store.Search([0f, 4f], topK: 1)[0].Score, 5);
        }

        [Fact]
        public void Search_TopK_LimitsResults()
        {
            var store = new VectorStore(dimension: 2);
            for (var i = 0; i < 5; i++)
            {
                store.Add($"v{i}", [i + 1f, 0f]);
            }

            Assert.Equal(2, store.Search([1f, 0f], topK: 2).Length);
            Assert.Equal(5, store.Search([1f, 0f], topK: 50).Length);   // clamped to Count
        }

        [Fact]
        public void Add_GrowsBeyondInitialCapacity()
        {
            var store = new VectorStore(dimension: 4, initialCapacity: 2);
            // Distinct directions (cosine ignores magnitude): v_i points along [1, i].
            for (var i = 0; i < 10; i++)
            {
                store.Add($"v{i}", [1f, i, 0f, 0f]);
            }
            Assert.Equal(10, store.Count);
            Assert.Equal("v9", store.Search([1f, 9f, 0f, 0f], topK: 1)[0].Id);   // matches v9's direction
        }

        [Fact]
        public void Add_DimensionMismatch_Throws()
        {
            var store = new VectorStore(dimension: 3);
            Assert.Throws<ArgumentException>(() => store.Add("bad", [1f, 0f]));
        }

        [Fact]
        public void Search_EmptyStore_ReturnsNothing()
        {
            var store = new VectorStore(dimension: 3);
            var buf = new VectorMatch[4];
            Assert.Equal(0, store.Search([1f, 0f, 0f], buf));
        }

        [Fact]
        public void SaveLoad_RoundTrips_VectorsIdsPayloads_AndSearch()
        {
            var store = new VectorStore(dimension: 3);
            store.Add("a", [3f, 0f, 0f], "alpha");          // un-normalised input — store normalises
            store.Add("b", [0f, 5f, 0f], null);             // null payload survives
            store.Add("c", [0f, 0f, 1f], "gamma");

            var path = Path.Combine(Path.GetTempPath(), $"ovs-{Guid.NewGuid():N}.bin");
            try
            {
                store.Save(path);
                var loaded = VectorStore.Load(path);

                Assert.Equal(store.Dimension, loaded.Dimension);
                Assert.Equal(store.Count, loaded.Count);

                for (var i = 0; i < store.Count; i++)
                {
                    Assert.Equal(store.GetId(i), loaded.GetId(i));
                    Assert.Equal(store.GetPayload(i), loaded.GetPayload(i));
                    // Vectors were stored unit-normalised; the reload is byte-verbatim.
                    Assert.True(store.GetVector(i).SequenceEqual(loaded.GetVector(i)));
                }

                // Search behaves identically after a reload (the index-once-restart-query guarantee).
                var before = store.Search([1f, 0f, 0f], topK: 3);
                var after = loaded.Search([1f, 0f, 0f], topK: 3);
                Assert.Equal(before.Length, after.Length);
                for (var i = 0; i < before.Length; i++)
                {
                    Assert.Equal(before[i].Id, after[i].Id);
                    Assert.Equal(before[i].Score, after[i].Score, precision: 6);
                }
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void SaveLoad_EmptyStore_RoundTrips()
        {
            var store = new VectorStore(dimension: 8);
            var path = Path.Combine(Path.GetTempPath(), $"ovs-{Guid.NewGuid():N}.bin");
            try
            {
                store.Save(path);
                var loaded = VectorStore.Load(path);
                Assert.Equal(8, loaded.Dimension);
                Assert.Equal(0, loaded.Count);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_NonVectorStoreFile_Throws()
        {
            var path = Path.Combine(Path.GetTempPath(), $"ovs-{Guid.NewGuid():N}.bin");
            try
            {
                File.WriteAllBytes(path, [1, 2, 3, 4, 5, 6, 7, 8]);
                Assert.Throws<OverfitFormatException>(() => VectorStore.Load(path));
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
