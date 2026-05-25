// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// An in-process, zero-dependency vector store: add embedding vectors (e.g. from
    /// <c>CachedLlamaSession.Embed</c>) and retrieve the top-K by cosine similarity. The whole RAG
    /// retrieval step stays inside your .NET process — no external vector database, no network.
    ///
    /// Vectors are stored unit-normalised in one contiguous backing array, so search is a flat
    /// dot-product scan with a top-K insertion pass (no full sort, no per-search allocation in the
    /// span overload). Linear scan — built for the thousands-to-low-millions of chunks a single
    /// app/document set produces, not a billion-scale ANN index.
    ///
    /// Not thread-safe for concurrent <see cref="Add"/>; concurrent reads are fine once populated.
    /// </summary>
    public sealed class VectorStore
    {
        private const float NormEpsilon = 1e-12f;

        private float[] _vectors;       // [Capacity * Dimension], unit-normalised, row per item
        private string[] _ids;
        private string?[] _payloads;

        public VectorStore(int dimension, int initialCapacity = 16)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(initialCapacity);

            Dimension = dimension;
            _vectors = new float[(long)initialCapacity * dimension <= int.MaxValue
                ? initialCapacity * dimension
                : throw new ArgumentOutOfRangeException(nameof(initialCapacity))];
            _ids = new string[initialCapacity];
            _payloads = new string?[initialCapacity];
        }

        /// <summary>Embedding dimension every added vector must match.</summary>
        public int Dimension { get; }

        /// <summary>Number of stored vectors.</summary>
        public int Count { get; private set; }

        /// <summary>
        /// Adds a vector under <paramref name="id"/> with an optional <paramref name="payload"/>
        /// (e.g. the source text). The vector is copied and unit-normalised; the caller's span is
        /// not retained.
        /// </summary>
        public void Add(string id, ReadOnlySpan<float> vector, string? payload = null)
        {
            ArgumentNullException.ThrowIfNull(id);
            if (vector.Length != Dimension)
            {
                throw new ArgumentException(
                    $"Vector length ({vector.Length}) must equal the store dimension ({Dimension}).",
                    nameof(vector));
            }

            EnsureCapacity(Count + 1);

            var dst = _vectors.AsSpan(Count * Dimension, Dimension);
            vector.CopyTo(dst);
            Normalize(dst);

            _ids[Count] = id;
            _payloads[Count] = payload;
            Count++;
        }

        /// <summary>
        /// Fills <paramref name="results"/> with the highest-cosine matches to <paramref name="query"/>,
        /// best first, and returns how many were written (≤ <c>results.Length</c> and ≤ <see cref="Count"/>).
        /// Zero-allocation. <see cref="VectorMatch.Score"/> is the true cosine similarity.
        /// </summary>
        public int Search(ReadOnlySpan<float> query, Span<VectorMatch> results)
        {
            if (query.Length != Dimension)
            {
                throw new ArgumentException(
                    $"Query length ({query.Length}) must equal the store dimension ({Dimension}).",
                    nameof(query));
            }

            var k = results.Length;
            if (k == 0 || Count == 0) { return 0; }

            // Stored vectors are unit-norm, so cosine = dot(query, v) / ‖query‖. ‖query‖ is constant
            // across items, so it doesn't change ranking — apply it only to report the true cosine.
            var queryNorm = MathF.Sqrt(Dot(query, query));
            var inverseNorm = queryNorm > NormEpsilon ? 1f / queryNorm : 0f;

            var found = 0;
            for (var i = 0; i < Count; i++)
            {
                var score = Dot(query, _vectors.AsSpan(i * Dimension, Dimension)) * inverseNorm;
                InsertDescending(results, ref found, k, new VectorMatch(_ids[i], score, _payloads[i]));
            }
            return found;
        }

        /// <summary>Convenience overload: allocates and returns up to <paramref name="topK"/> matches.</summary>
        public VectorMatch[] Search(ReadOnlySpan<float> query, int topK)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);
            var capacity = Math.Min(topK, Count);
            if (capacity == 0) { return []; }

            var buffer = new VectorMatch[capacity];
            var written = Search(query, buffer);
            return written == buffer.Length ? buffer : buffer[..written];
        }

        private static void InsertDescending(Span<VectorMatch> results, ref int found, int k, VectorMatch candidate)
        {
            // Reject early if the list is full and the candidate can't beat the current worst.
            if (found == k && candidate.Score <= results[k - 1].Score) { return; }

            var pos = found < k ? found : k - 1;
            while (pos > 0 && results[pos - 1].Score < candidate.Score)
            {
                results[pos] = results[pos - 1];
                pos--;
            }
            results[pos] = candidate;
            if (found < k) { found++; }
        }

        private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            var sum = 0f;
            for (var i = 0; i < a.Length; i++) { sum += a[i] * b[i]; }
            return sum;
        }

        private static void Normalize(Span<float> v)
        {
            var norm = MathF.Sqrt(Dot(v, v));
            if (norm <= NormEpsilon) { return; }
            var inv = 1f / norm;
            for (var i = 0; i < v.Length; i++) { v[i] *= inv; }
        }

        private void EnsureCapacity(int needed)
        {
            var capacity = _ids.Length;
            if (needed <= capacity) { return; }

            var newCapacity = capacity * 2;
            while (newCapacity < needed) { newCapacity *= 2; }

            Array.Resize(ref _vectors, newCapacity * Dimension);
            Array.Resize(ref _ids, newCapacity);
            Array.Resize(ref _payloads, newCapacity);
        }
    }
}
