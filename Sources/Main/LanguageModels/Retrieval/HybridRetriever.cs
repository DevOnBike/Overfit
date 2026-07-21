// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// Hybrid retrieval: a <see cref="VectorStore"/> (semantic) and a <see cref="Bm25Index"/> (lexical) kept
    /// in lock-step over one corpus, queried together and merged with <see cref="ReciprocalRankFusion"/>.
    ///
    /// <para>The two arms fail in opposite directions, which is the entire point. Dense search finds
    /// "how do I cancel my policy" in a chunk that says "termination of cover" and never uses the word
    /// cancel. Lexical search finds policy number <c>PL-88-40021</c>, which dense search cannot distinguish
    /// from <c>PL-88-40022</c>. Neither is a superset of the other, so a corpus containing both prose and
    /// identifiers — i.e. essentially every real enterprise corpus — needs both.</para>
    ///
    /// <para>Both indexes are populated through this type's <see cref="Add"/>, so they cannot drift out of
    /// sync. If you need one arm alone, <see cref="Vectors"/> and <see cref="Lexical"/> expose them
    /// directly — useful for measuring what hybrid actually bought you on your own corpus rather than
    /// assuming it helped.</para>
    /// </summary>
    public sealed class HybridRetriever
    {
        private readonly VectorStore _vectors;
        private readonly Bm25Index _lexical;

        public HybridRetriever(int dimension, int initialCapacity = 16)
        {
            _vectors = new VectorStore(dimension, initialCapacity);
            _lexical = new Bm25Index();
        }

        /// <summary>Wraps an existing pair of indexes (e.g. a <see cref="VectorStore"/> reloaded from disk).
        /// The caller is responsible for them describing the same corpus.</summary>
        public HybridRetriever(VectorStore vectors, Bm25Index lexical)
        {
            ArgumentNullException.ThrowIfNull(vectors);
            ArgumentNullException.ThrowIfNull(lexical);

            _vectors = vectors;
            _lexical = lexical;
        }

        /// <summary>The semantic arm.</summary>
        public VectorStore Vectors => _vectors;

        /// <summary>The lexical arm.</summary>
        public Bm25Index Lexical => _lexical;

        /// <summary>Number of indexed chunks.</summary>
        public int Count => _vectors.Count;

        /// <summary>
        /// Indexes one chunk into both arms: <paramref name="vector"/> into the vector store and
        /// <paramref name="text"/> into the BM25 index. <paramref name="payload"/> defaults to
        /// <paramref name="text"/>, since a retrieved chunk almost always needs its text to build the prompt.
        /// </summary>
        public void Add(string id, ReadOnlySpan<float> vector, string text, string? payload = null)
        {
            ArgumentNullException.ThrowIfNull(id);
            ArgumentNullException.ThrowIfNull(text);

            var effectivePayload = payload ?? text;
            _vectors.Add(id, vector, effectivePayload);
            _lexical.Add(id, text, effectivePayload);
        }

        /// <summary>
        /// Retrieves the top-<paramref name="topK"/> chunks for a query given both its embedding and its raw
        /// text.
        ///
        /// <para><paramref name="candidatesPerArm"/> is the depth each arm is asked for before fusion; it
        /// defaults to <c>max(4·topK, 20)</c>. Fusion needs headroom to work — if each arm returns only
        /// <c>topK</c>, a document ranked just outside both lists can never be promoted by agreement, which
        /// is the effect hybrid retrieval exists to capture. Deeper costs almost nothing here because both
        /// arms already scan the corpus; only the merge grows.</para>
        /// </summary>
        public VectorMatch[] Search(
            ReadOnlySpan<float> queryVector,
            string queryText,
            int topK,
            int candidatesPerArm = 0)
        {
            ArgumentNullException.ThrowIfNull(queryText);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

            if (Count == 0)
            {
                return [];
            }

            var depth = candidatesPerArm > 0 ? candidatesPerArm : Math.Max(4 * topK, 20);
            depth = Math.Min(depth, Count);

            var dense = new VectorMatch[depth];
            var denseCount = _vectors.Search(queryVector, dense);

            var lexical = new VectorMatch[depth];
            var lexicalCount = _lexical.Search(queryText, lexical);

            var results = new VectorMatch[Math.Min(topK, Count)];
            var written = ReciprocalRankFusion.Fuse(
                dense.AsSpan(0, denseCount), lexical.AsSpan(0, lexicalCount), results);

            return written == results.Length ? results : results[..written];
        }
    }
}
