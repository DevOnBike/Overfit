// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// The production RAG path: persist a collection (index once), reload without re-embedding, and re-index
    /// only changed source documents (content-hash manifest). Pure-managed — no SQLite, no native dependency.
    /// </summary>
    public sealed class PersistentVectorStoreTests
    {
        private static VectorChunk Chunk(string id, float[] v, string? payload = null) => new(id, v, payload);

        [Fact]
        public void NeedsReindex_TracksNewChangedAndUnchangedSources()
        {
            var store = new PersistentVectorStore(dimension: 3, collectionName: "docs");

            Assert.True(store.NeedsReindex("a.md", "hash1"));            // never seen → reindex

            store.IndexSource("a.md", "hash1", [Chunk("a#0", [1f, 0f, 0f])]);
            Assert.False(store.NeedsReindex("a.md", "hash1"));           // same hash → skip
            Assert.True(store.NeedsReindex("a.md", "hash2"));            // changed → reindex
            Assert.True(store.NeedsReindex("b.md", "hash1"));            // different source → reindex
        }

        [Fact]
        public void IndexSource_ChangedHash_ReplacesOldChunks()
        {
            var store = new PersistentVectorStore(dimension: 3, collectionName: "docs");
            store.IndexSource("a.md", "v1", [Chunk("a#0", [1f, 0f, 0f]), Chunk("a#1", [0f, 1f, 0f])]);
            store.IndexSource("b.md", "v1", [Chunk("b#0", [0f, 0f, 1f])]);
            Assert.Equal(3, store.Count);
            Assert.Equal(2, store.SourceCount);

            // a.md changed → re-index with a single (different) chunk; b.md untouched.
            store.IndexSource("a.md", "v2", [Chunk("a#0", [1f, 1f, 0f])]);
            Assert.Equal(2, store.Count);                                 // 1 (new a) + 1 (b)
            Assert.Equal(2, store.SourceCount);
            Assert.False(store.NeedsReindex("a.md", "v2"));
            Assert.False(store.NeedsReindex("b.md", "v1"));               // b survived the a re-index

            // b.md is still retrievable after a's rebuild.
            Assert.Equal("b#0", store.Search([0f, 0f, 1f], topK: 1)[0].Id);
        }

        [Fact]
        public void RemoveSource_DropsOnlyThatSource()
        {
            var store = new PersistentVectorStore(dimension: 2, collectionName: "docs");
            store.IndexSource("a.md", "h", [Chunk("a#0", [1f, 0f])]);
            store.IndexSource("b.md", "h", [Chunk("b#0", [0f, 1f])]);

            store.RemoveSource("a.md");
            Assert.Equal(1, store.Count);
            Assert.Equal(1, store.SourceCount);
            Assert.True(store.NeedsReindex("a.md", "h"));                 // gone
            Assert.Equal("b#0", store.Search([0f, 1f], topK: 1)[0].Id);
        }

        [Fact]
        public void SaveLoad_RoundTrips_Collection_Manifest_AndSearch()
        {
            var store = new PersistentVectorStore(dimension: 3, collectionName: "policies");
            store.IndexSource("a.md", "h1", [Chunk("a#0", [2f, 0f, 0f], "alpha"), Chunk("a#1", [0f, 2f, 0f], null)]);
            store.IndexSource("b.md", "h2", [Chunk("b#0", [0f, 0f, 9f], "gamma")]);

            var path = Path.Combine(Path.GetTempPath(), $"psp-{Guid.NewGuid():N}.bin");
            try
            {
                store.Save(path);
                var loaded = PersistentVectorStore.Load(path);

                Assert.Equal("policies", loaded.CollectionName);
                Assert.Equal(store.Count, loaded.Count);
                Assert.Equal(store.SourceCount, loaded.SourceCount);

                // Manifest survived → unchanged sources are still skippable after a restart (no re-embed).
                Assert.False(loaded.NeedsReindex("a.md", "h1"));
                Assert.False(loaded.NeedsReindex("b.md", "h2"));
                Assert.True(loaded.NeedsReindex("a.md", "h-different"));

                // Retrieval is identical after the reload.
                Assert.Equal("a#0", loaded.Search([1f, 0f, 0f], topK: 1)[0].Id);
                Assert.Equal("alpha", loaded.Store.GetPayload(0));
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
