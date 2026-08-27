// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// A <see cref="VectorStore"/> with on-disk persistence and a source-document manifest — the production
    /// RAG path: <b>index once, restart the service, query without re-embedding</b>, and on a re-run only
    /// re-embed the documents that actually changed. Pure-managed (one binary file via <see cref="VectorStore"/>'s
    /// serializer + a small manifest) — no SQLite, no native dependency, so the Native-AOT / no-native-binary
    /// identity holds.
    ///
    /// <para>
    /// Each source document is registered under a stable <c>sourceId</c> (e.g. its path) with a
    /// <c>contentHash</c>. <see cref="NeedsReindex"/> tells the caller whether a source is new or changed so it
    /// can skip embedding the unchanged ones; <see cref="IndexSource"/> (re)indexes one source, replacing its
    /// previous chunks if its hash moved. Retrieval delegates to the underlying flat-scan store (built for
    /// thousands-to-low-millions of chunks, not billion-scale ANN).
    /// </para>
    /// </summary>
    public sealed class PersistentVectorStore
    {
        private const uint FileMagic = 0x3150_5350; // "PSP1" little-endian (Persistent Store, v1)

        // 1 -> 2 on 2026-08-27 (XC-131). Version 1 files carry no embedding-space id, so they cannot be
        // checked against one and must be rejected. That is the point of the bump rather than a side effect:
        // XC-131 changed what CachedLlamaSession.Embed returns, and a v1 cache written before it reloads
        // cleanly afterwards — same dimension, same source count, same content hashes — after which every
        // query cosine is computed between two different embedding spaces, with nothing thrown and nothing
        // logged. There is no reader for v1 and none is planned; a cache is reproducible from its sources.
        private const int FileVersion = 2;

        private VectorStore _store;
        private readonly Dictionary<string, SourceEntry> _sources;

        /// <param name="dimension">Vector width.</param>
        /// <param name="collectionName">A human label for this index. Persisted.</param>
        /// <param name="embeddingSpaceId">Identifies the embedding space these vectors live in; persisted and
        /// checked on <see cref="Load"/>. See <see cref="EmbeddingSpaceId"/> for what belongs in it.</param>
        public PersistentVectorStore(int dimension, string collectionName = "default", string embeddingSpaceId = "")
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
            ArgumentException.ThrowIfNullOrEmpty(collectionName);
            ArgumentNullException.ThrowIfNull(embeddingSpaceId);

            _store = new VectorStore(dimension);
            _sources = new Dictionary<string, SourceEntry>(StringComparer.Ordinal);
            CollectionName = collectionName;
            EmbeddingSpaceId = embeddingSpaceId;
        }

        private PersistentVectorStore(
            string collectionName,
            string embeddingSpaceId,
            VectorStore store,
            Dictionary<string, SourceEntry> sources)
        {
            CollectionName = collectionName;
            EmbeddingSpaceId = embeddingSpaceId;
            _store = store;
            _sources = sources;
        }

        /// <summary>A human label for this index (e.g. the corpus / tenant name). Persisted.</summary>
        public string CollectionName
        {
            get;
        }

        /// <summary>
        /// Opaque identity of the embedding space these vectors live in. Persisted, and
        /// <see cref="Load"/> refuses a file whose id differs from the one the caller expects.
        ///
        /// <para><b>The store cannot compute this and does not try.</b> It receives <c>float[]</c> and has no
        /// idea which model, pooling mode, prefix or quantisation produced them — only the caller does. What
        /// to put in it: everything whose change would move the vectors. For a GGUF embedder that is the
        /// model file's identity, the pooling mode, the query and passage prefixes, and the quantise flag.</para>
        ///
        /// <para><b>Empty means "unknown" and does not match a non-empty id.</b> A wildcard would restore the
        /// silent cross-space comparison this field exists to reject, so the comparison is plain equality and
        /// the field only earns its keep once callers populate it.</para>
        /// </summary>
        public string EmbeddingSpaceId
        {
            get;
        }

        public int Dimension => _store.Dimension;

        /// <summary>Total chunks across all sources.</summary>
        public int Count => _store.Count;

        /// <summary>Number of distinct source documents indexed.</summary>
        public int SourceCount => _sources.Count;

        /// <summary>The source ids currently indexed.</summary>
        public IReadOnlyCollection<string> SourceIds => _sources.Keys;

        /// <summary>How many chunks the given source contributed, or 0 if it is not indexed.</summary>
        public int GetSourceChunkCount(string sourceId)
        {
            ArgumentNullException.ThrowIfNull(sourceId);
            return _sources.TryGetValue(sourceId, out var entry) ? entry.Ids.Length : 0;
        }

        /// <summary>The underlying vector store, for retrieval / corpus analysis.</summary>
        public VectorStore Store => _store;

        /// <summary>
        /// True when <paramref name="sourceId"/> has not been indexed, or its content has changed since it was
        /// (its stored <c>contentHash</c> differs). The caller embeds + <see cref="IndexSource"/> only when this
        /// returns true — the "re-embed only changed files" win.
        /// </summary>
        public bool NeedsReindex(string sourceId, string contentHash)
        {
            ArgumentNullException.ThrowIfNull(sourceId);
            ArgumentNullException.ThrowIfNull(contentHash);
            return !_sources.TryGetValue(sourceId, out var entry)
                || !string.Equals(entry.Hash, contentHash, StringComparison.Ordinal);
        }

        /// <summary>
        /// Indexes (or re-indexes) one source document: drops the source's previous chunks if it was already
        /// present (a changed file), adds <paramref name="chunks"/>, and records <paramref name="contentHash"/>
        /// so a future run can <see cref="NeedsReindex"/>-skip it when unchanged. Chunk ids should be stable
        /// per (source, chunk) so retrieval results are reproducible.
        /// </summary>
        public void IndexSource(
            string sourceId,
            string contentHash,
            IReadOnlyList<VectorChunk> chunks)
        {
            ArgumentNullException.ThrowIfNull(sourceId);
            ArgumentNullException.ThrowIfNull(contentHash);
            ArgumentNullException.ThrowIfNull(chunks);

            if (_sources.ContainsKey(sourceId))
            {
                RemoveSource(sourceId);
            }

            var ids = new string[chunks.Count];
            for (var i = 0; i < chunks.Count; i++)
            {
                var chunk = chunks[i];
                _store.Add(chunk.Id, chunk.Vector, chunk.Payload);
                ids[i] = chunk.Id;
            }

            _sources[sourceId] = new SourceEntry(contentHash, ids);
        }

        /// <summary>
        /// Removes a source document and all its chunks. The flat store is append-only, so this rebuilds it
        /// without the removed chunks (cheap in memory — the cost is re-embedding, which this avoids). No-op if
        /// the source is not present.
        /// </summary>
        public void RemoveSource(string sourceId)
        {
            ArgumentNullException.ThrowIfNull(sourceId);
            if (!_sources.Remove(sourceId, out var removed))
            {
                return;
            }

            var dropped = new HashSet<string>(removed.Ids, StringComparer.Ordinal);
            var rebuilt = new VectorStore(_store.Dimension, Math.Max(_store.Count, 1));
            for (var i = 0; i < _store.Count; i++)
            {
                var id = _store.GetId(i);
                if (!dropped.Contains(id))
                {
                    rebuilt.Add(id, _store.GetVector(i), _store.GetPayload(i));
                }
            }
            _store = rebuilt;
        }

        /// <summary>Top-K retrieval — delegates to the underlying <see cref="VectorStore"/>.</summary>
        public VectorMatch[] Search(ReadOnlySpan<float> query, int topK) => _store.Search(query, topK);

        /// <summary>Zero-allocation top-K retrieval — delegates to the underlying <see cref="VectorStore"/>.</summary>
        public int Search(ReadOnlySpan<float> query, Span<VectorMatch> results) => _store.Search(query, results);

        /// <summary>Writes the collection (name + source manifest + all vectors) to one binary file.</summary>
        public void Save(string path)
        {
            ArgumentNullException.ThrowIfNull(path);

            using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(stream);

            writer.Write(FileMagic);
            writer.Write(FileVersion);
            writer.Write(CollectionName);
            writer.Write(EmbeddingSpaceId);

            writer.Write(_sources.Count);
            foreach (var (sourceId, entry) in _sources)
            {
                writer.Write(sourceId);
                writer.Write(entry.Hash);
                writer.Write(entry.Ids.Length);
                foreach (var id in entry.Ids)
                {
                    writer.Write(id);
                }
            }

            _store.WriteTo(writer);
        }

        /// <summary>
        /// Reloads a collection written by <see cref="Save"/> — index-once-restart-query.
        /// </summary>
        /// <param name="path">The file written by <see cref="Save"/>.</param>
        /// <param name="expectedEmbeddingSpaceId">The space the caller is about to compare against. A file
        /// written in a different space throws <see cref="OverfitFormatException"/> rather than loading,
        /// because a cosine across two embedding spaces is meaningless and nothing downstream can detect it.
        /// See <see cref="EmbeddingSpaceId"/>.</param>
        public static PersistentVectorStore Load(string path, string expectedEmbeddingSpaceId = "")
        {
            ArgumentNullException.ThrowIfNull(path);
            ArgumentNullException.ThrowIfNull(expectedEmbeddingSpaceId);

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(stream);

            if (reader.ReadUInt32() != FileMagic)
            {
                throw new OverfitFormatException($"'{path}' is not an Overfit persistent-vector-store file (bad magic).");
            }
            var version = reader.ReadInt32();
            if (version != FileVersion)
            {
                throw new OverfitFormatException($"Unsupported persistent-vector-store version {version} (expected {FileVersion}).");
            }

            var collectionName = reader.ReadString();
            var embeddingSpaceId = reader.ReadString();
            if (!string.Equals(embeddingSpaceId, expectedEmbeddingSpaceId, StringComparison.Ordinal))
            {
                throw new OverfitFormatException(
                    $"'{path}' was written in embedding space '{embeddingSpaceId}' but the caller expects "
                    + $"'{expectedEmbeddingSpaceId}'. The stored vectors cannot be compared against vectors "
                    + "from a different space; re-index the corpus.");
            }

            var sourceCount = reader.ReadInt32();
            if (sourceCount < 0)
            {
                throw new OverfitFormatException($"Corrupt source manifest (count {sourceCount}).");
            }

            var sources = new Dictionary<string, SourceEntry>(sourceCount, StringComparer.Ordinal);
            for (var s = 0; s < sourceCount; s++)
            {
                var sourceId = reader.ReadString();
                var hash = reader.ReadString();
                var idCount = reader.ReadInt32();
                if (idCount < 0)
                {
                    throw new OverfitFormatException($"Corrupt source manifest (id count {idCount}).");
                }
                var ids = new string[idCount];
                for (var i = 0; i < idCount; i++)
                {
                    ids[i] = reader.ReadString();
                }
                sources[sourceId] = new SourceEntry(hash, ids);
            }

            var store = VectorStore.ReadFrom(reader);
            return new PersistentVectorStore(collectionName, embeddingSpaceId, store, sources);
        }

        /// <summary>One indexed source document: its content hash + the chunk ids it produced.</summary>
        private readonly struct SourceEntry
        {
            public SourceEntry(string hash, string[] ids)
            {
                Hash = hash;
                Ids = ids;
            }

            public string Hash
            {
                get;
            }

            public string[] Ids
            {
                get;
            }
        }
    }
}
