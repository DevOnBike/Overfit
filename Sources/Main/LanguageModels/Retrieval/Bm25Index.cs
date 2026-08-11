// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// Okapi BM25 lexical index — the keyword half of hybrid retrieval, and the half a
    /// <see cref="VectorStore"/> is structurally bad at.
    ///
    /// <para><b>Why this exists.</b> Dense embeddings match on <i>meaning</i>, which is exactly wrong for the
    /// tokens enterprise corpora are full of: case numbers, policy numbers, part numbers, ICD codes, tax ids,
    /// error codes. "II CSK 345/21" and "II CSK 346/21" are near-identical directions in embedding space and
    /// completely different documents in reality. BM25 matches on the literal term, so it retrieves them
    /// exactly. Fuse the two with <see cref="ReciprocalRankFusion"/> and you get both.</para>
    ///
    /// <para>Pure algorithm: no second model, no extra weights, no GPU, no network — it is a dictionary and
    /// some arithmetic, so it holds the engine's on-prem / Native-AOT / no-native-dependency identity. Search
    /// accumulates into a pooled score buffer and does a top-K insertion pass rather than sorting the corpus.</para>
    ///
    /// <para>Linear scan over the postings of the query's terms — built for the thousands-to-low-millions of
    /// chunks one document set produces, matching <see cref="VectorStore"/>'s scale. Not thread-safe for
    /// concurrent <see cref="Add"/>; concurrent reads are fine once populated.</para>
    /// </summary>
    public sealed class Bm25Index
    {
        /// <summary>Term-frequency saturation. Standard Okapi default; higher = term repetition keeps mattering longer.</summary>
        public const float DefaultK1 = 1.2f;

        /// <summary>Document-length normalisation strength, in [0,1]. 0 = ignore length, 1 = fully normalise.</summary>
        public const float DefaultB = 0.75f;

        private readonly Dictionary<string, int> _termIds = new(StringComparer.Ordinal);
        private readonly List<List<Posting>> _postings = [];
        private readonly List<string> _ids = [];
        private readonly List<string?> _payloads = [];
        private readonly List<int> _documentLengths = [];
        private long _totalTokens;

        public Bm25Index(float k1 = DefaultK1, float b = DefaultB)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(k1);
            ArgumentOutOfRangeException.ThrowIfLessThan(b, 0f);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(b, 1f);

            K1 = k1;
            B = b;
        }

        /// <summary>Number of indexed documents.</summary>
        public int Count => _ids.Count;

        /// <summary>Number of distinct terms seen across the corpus.</summary>
        public int TermCount => _postings.Count;

        public float K1
        {
            get;
        }

        public float B
        {
            get;
        }

        /// <summary>
        /// Indexes <paramref name="text"/> under <paramref name="id"/> with an optional
        /// <paramref name="payload"/> (usually the source text itself, so a hit can be handed straight to a
        /// prompt). Only term frequencies are retained — the text is not stored unless it is the payload.
        /// </summary>
        public void Add(string id, string text, string? payload = null)
        {
            ArgumentNullException.ThrowIfNull(id);
            ArgumentNullException.ThrowIfNull(text);

            var documentIndex = _ids.Count;
            var tokens = Tokenize(text);

            // Collapse to per-term counts first: one posting per (document, term), not per occurrence.
            var counts = new Dictionary<int, int>();
            for (var i = 0; i < tokens.Count; i++)
            {
                var termId = InternTerm(tokens[i]);
                counts.TryGetValue(termId, out var current);
                counts[termId] = current + 1;
            }

            foreach (var pair in counts)
            {
                _postings[pair.Key].Add(new Posting(documentIndex, pair.Value));
            }

            _ids.Add(id);
            _payloads.Add(payload);
            _documentLengths.Add(tokens.Count);
            _totalTokens += tokens.Count;
        }

        /// <summary>
        /// Fills <paramref name="results"/> with the best BM25 matches for <paramref name="query"/>, best
        /// first, and returns how many were written. Documents scoring zero (no query term present) are never
        /// returned, so this can write fewer than <c>results.Length</c> even on a large corpus.
        ///
        /// <para><see cref="VectorMatch.Score"/> carries the BM25 score here, <b>not</b> a cosine — BM25 is
        /// unbounded and corpus-relative, so it is meaningful for ranking and meaningless as an absolute
        /// threshold. This is precisely why <see cref="ReciprocalRankFusion"/> fuses <i>ranks</i> rather than
        /// scores: the two arms never need their scales reconciled.</para>
        /// </summary>
        public int Search(string query, Span<VectorMatch> results)
        {
            ArgumentNullException.ThrowIfNull(query);

            var k = results.Length;
            if (k == 0 || Count == 0)
            {
                return 0;
            }

            var queryTokens = Tokenize(query);
            if (queryTokens.Count == 0)
            {
                return 0;
            }

            using var scoreBuffer = new PooledBuffer<float>(Count, clearMemory: true);
            var scores = scoreBuffer.Span.Slice(0, Count);

            var averageLength = (float)((double)_totalTokens / Count);
            var scored = new HashSet<int>();

            for (var t = 0; t < queryTokens.Count; t++)
            {
                if (!_termIds.TryGetValue(queryTokens[t], out var termId))
                {
                    continue; // term absent from the corpus — contributes nothing
                }

                // A term repeated in the query must not count twice: BM25 saturates term frequency in the
                // DOCUMENT, and query-side repetition is not evidence about the document.
                if (!scored.Add(termId))
                {
                    continue;
                }

                var postings = _postings[termId];
                var documentFrequency = postings.Count;
                var idf = MathF.Log(1f + (Count - documentFrequency + 0.5f) / (documentFrequency + 0.5f));

                for (var p = 0; p < postings.Count; p++)
                {
                    var posting = postings[p];
                    var termFrequency = (float)posting.Frequency;
                    var lengthNorm = 1f - B + (B * _documentLengths[posting.DocumentIndex] / averageLength);
                    scores[posting.DocumentIndex] +=
                        idf * (termFrequency * (K1 + 1f)) / (termFrequency + (K1 * lengthNorm));
                }
            }

            var found = 0;
            for (var i = 0; i < Count; i++)
            {
                if (scores[i] <= 0f)
                {
                    continue; // untouched by any query term — not a match at all
                }

                TopKMatchSelector.InsertDescending(
                    results, ref found, k, new VectorMatch(_ids[i], scores[i], _payloads[i]));
            }

            return found;
        }

        /// <summary>Convenience overload: allocates and returns up to <paramref name="topK"/> matches.</summary>
        public VectorMatch[] Search(string query, int topK)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

            var capacity = Math.Min(topK, Count);
            if (capacity == 0)
            {
                return [];
            }

            var buffer = new VectorMatch[capacity];
            var written = Search(query, buffer);
            return written == buffer.Length ? buffer : buffer.AsSpan(0, written).ToArray();
        }

        /// <summary>
        /// Splits <paramref name="text"/> into lower-cased alphanumeric terms — the shared tokenisation used
        /// for both indexing and querying, so the two can never drift apart.
        ///
        /// <para>Deliberately Unicode-aware via <see cref="char.IsLetterOrDigit(char)"/> rather than an ASCII
        /// range: Polish (ą, ć, ę, ł, ń, ó, ś, ź, ż) and every other non-ASCII alphabet must survive
        /// tokenisation, and an ASCII-only split would shred them into fragments. No stemming and no
        /// stop-word list — both are language-specific, and getting them wrong costs more recall than the
        /// index size they save.</para>
        ///
        /// <para><b>Connector-joined runs emit the whole form as an extra term.</b> <c>OVERFIT_DECODE_WORKERS</c>
        /// yields <c>overfit</c>, <c>decode</c>, <c>workers</c> AND <c>overfit_decode_workers</c>. Splitting
        /// alone was measured to lose exactly this case: the three parts are among the commonest words in a
        /// .NET corpus, so their IDF is near zero and the document defining the variable was outranked, while
        /// <c>block_q4_Kx8</c> survived only because <c>q4</c>/<c>kx8</c> happen to stay rare. The joined form
        /// is maximally rare, so it carries the IDF the parts cannot. Keeping the parts too means a query for
        /// one component still matches.</para>
        ///
        /// <para>The joined term is emitted <b>only</b> when the run really has two or more parts — emitting it
        /// for ordinary words would double their term frequency and corrupt both the TF saturation and the
        /// document-length normalisation.</para>
        /// </summary>
        public static List<string> Tokenize(string text)
        {
            ArgumentNullException.ThrowIfNull(text);

            var tokens = new List<string>();
            var i = 0;

            // Bounded by text.Length; every path through the body advances `i` at least once.
            while (i < text.Length)
            {
                if (!char.IsLetterOrDigit(text[i]))
                {
                    i++;
                    continue;
                }

                var runStart = i;
                var partStart = i;
                var parts = 0;

                while (i < text.Length)
                {
                    if (char.IsLetterOrDigit(text[i]))
                    {
                        i++;
                        continue;
                    }

                    // A connector only continues the run when it sits BETWEEN two alphanumerics, so a trailing
                    // hyphen or an em-dash between words still terminates it.
                    var isConnector = (text[i] == '_' || text[i] == '-')
                        && i + 1 < text.Length
                        && char.IsLetterOrDigit(text[i + 1]);

                    if (!isConnector)
                    {
                        break;
                    }

                    tokens.Add(Lower(text, partStart, i - partStart));
                    parts++;
                    i++;
                    partStart = i;
                }

                tokens.Add(Lower(text, partStart, i - partStart));
                parts++;

                if (parts > 1)
                {
                    tokens.Add(Lower(text, runStart, i - runStart));
                }
            }

            return tokens;
        }

        private static string Lower(string text, int start, int length)
            => text.AsSpan(start, length).ToString().ToLowerInvariant();

        private int InternTerm(string term)
        {
            if (_termIds.TryGetValue(term, out var existing))
            {
                return existing;
            }

            var termId = _postings.Count;
            _termIds[term] = termId;
            _postings.Add([]);
            return termId;
        }

        /// <summary>One (document, term-frequency) entry in a term's postings list.</summary>
        private readonly struct Posting
        {
            public Posting(int documentIndex, int frequency)
            {
                DocumentIndex = documentIndex;
                Frequency = frequency;
            }

            public int DocumentIndex
            {
                get;
            }

            public int Frequency
            {
                get;
            }
        }
    }
}
