// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval
{
    /// <summary>
    /// Reciprocal Rank Fusion — merges several ranked result lists into one by summing <c>1 / (k + rank)</c>
    /// per document.
    ///
    /// <para><b>Why rank fusion and not score fusion.</b> The two retrieval arms produce incomparable numbers:
    /// <see cref="VectorStore"/> returns a cosine in [-1,1], <see cref="Bm25Index"/> returns an unbounded,
    /// corpus-relative BM25 score. Normalising them onto a common scale requires knowing each arm's score
    /// distribution, which shifts with the corpus and the query — a tuning knob that silently rots. Ranks have
    /// no such problem: position 1 means the same thing in both lists, forever. That is the whole reason RRF
    /// is the default fusion in practice despite being almost trivially simple.</para>
    ///
    /// <para>The constant <see cref="DefaultK"/> damps the top of each list: without it the rank-1 document of
    /// a single arm would dominate every fusion, so one confident-but-wrong arm could not be outvoted. At
    /// k=60 the gap between rank 1 and rank 2 is small enough that agreement across arms outranks depth
    /// within one arm — which is the behaviour hybrid retrieval is bought for.</para>
    /// </summary>
    public static class ReciprocalRankFusion
    {
        /// <summary>Rank-damping constant. 60 is the value from the original RRF paper and the de-facto default.</summary>
        public const float DefaultK = 60f;

        /// <summary>
        /// Fuses two ranked lists (each already best-first) into <paramref name="results"/>, best first, and
        /// returns how many were written. A document present in both lists accumulates both contributions,
        /// which is what lets agreement beat depth.
        ///
        /// <para><see cref="VectorMatch.Score"/> on the output is the fused RRF score — a small positive
        /// number with no meaning outside this comparison. Payloads are carried over from whichever input
        /// supplied a non-null one.</para>
        /// </summary>
        public static int Fuse(
            ReadOnlySpan<VectorMatch> first,
            ReadOnlySpan<VectorMatch> second,
            Span<VectorMatch> results,
            float k = DefaultK)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);

            if (results.Length == 0)
            {
                return 0;
            }

            var scores = new Dictionary<string, float>(StringComparer.Ordinal);
            var payloads = new Dictionary<string, string?>(StringComparer.Ordinal);

            Accumulate(first, scores, payloads, k);
            Accumulate(second, scores, payloads, k);

            var found = 0;
            foreach (var pair in scores)
            {
                payloads.TryGetValue(pair.Key, out var payload);
                TopKMatchSelector.InsertDescending(
                    results, ref found, results.Length, new VectorMatch(pair.Key, pair.Value, payload));
            }

            return found;
        }

        /// <summary>Convenience overload: allocates and returns up to <paramref name="topK"/> fused matches.</summary>
        public static VectorMatch[] Fuse(
            ReadOnlySpan<VectorMatch> first,
            ReadOnlySpan<VectorMatch> second,
            int topK,
            float k = DefaultK)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(topK);

            var buffer = new VectorMatch[topK];
            var written = Fuse(first, second, buffer, k);
            return written == buffer.Length ? buffer : buffer[..written];
        }

        private static void Accumulate(
            ReadOnlySpan<VectorMatch> ranked,
            Dictionary<string, float> scores,
            Dictionary<string, string?> payloads,
            float k)
        {
            for (var rank = 0; rank < ranked.Length; rank++)
            {
                var id = ranked[rank].Id;

                scores.TryGetValue(id, out var current);
                scores[id] = current + (1f / (k + rank + 1f)); // rank is 0-based here, 1-based in the formula

                if (ranked[rank].Payload is not null)
                {
                    payloads[id] = ranked[rank].Payload;
                }
            }
        }
    }
}
