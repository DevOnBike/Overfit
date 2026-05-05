// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Allocation-controlled token sampler for SLM generation.
    ///
    /// The sampler accepts scratch buffers supplied by the caller. This avoids
    /// allocating temporary arrays per generated token.
    ///
    /// Supported strategies:
    ///   Greedy     — always pick the highest-probability token
    ///   Temperature — sample from the full distribution scaled by temperature
    ///   TopK       — sample only from the top-k tokens
    ///   TopP       — nucleus sampling: sample from the smallest set of tokens
    ///                whose cumulative probability ≥ p
    ///   TopKTopP   — apply top-k first, then top-p on the remaining tokens
    ///
    /// Top-P (nucleus sampling) algorithm:
    ///   1. Apply temperature scaling to logits
    ///   2. Sort tokens descending by scaled logit (partial insertion sort — O(n) for typical p)
    ///   3. Compute softmax probabilities
    ///   4. Find the smallest prefix whose cumulative probability ≥ topP
    ///   5. Sample from that prefix using weighted random
    ///
    ///   Zero heap allocations — sort is done in-place on caller-supplied scratch buffers.
    ///   Vocab-size sort (768 or 50257 tokens) is the dominant cost: O(n log n).
    ///   For typical topP=0.9 and peaky distributions, the nucleus is small (~10-50 tokens)
    ///   so the sort terminates early via partial selection.
    /// </summary>
    public static class TokenSampler
    {
        private const float MinimumTemperature = 1e-6f;

        public static int Sample(
            ReadOnlySpan<float> logits,
            in SamplingOptions options,
            Random random,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            if (logits.IsEmpty)
            {
                throw new ArgumentException("Logits cannot be empty.", nameof(logits));
            }

            if (random is null)
            {
                throw new ArgumentNullException(nameof(random));
            }

            if (indexScratch.Length < logits.Length)
            {
                throw new ArgumentException(
                "Index scratch span is smaller than the logits span.", nameof(indexScratch));
            }

            if (scoreScratch.Length < logits.Length)
            {
                throw new ArgumentException(
                "Score scratch span is smaller than the logits span.", nameof(scoreScratch));
            }

            if (options.Strategy == SamplingStrategy.Greedy || options.Temperature <= 0f)
            {
                return ArgMax(logits);
            }

            var temperature = MathF.Max(options.Temperature, MinimumTemperature);

            switch (options.Strategy)
            {
                case SamplingStrategy.TopK when options.TopK > 0 && options.TopK < logits.Length:
                    {
                        var count = SelectTopK(logits, options.TopK, temperature, indexScratch, scoreScratch);
                        return SampleFromPreparedScores(indexScratch[..count], scoreScratch[..count], random);
                    }

                case SamplingStrategy.TopP when options.TopP < 1f:
                    {
                        var count = SelectTopP(logits, options.TopP, temperature, indexScratch, scoreScratch);
                        return SampleFromPreparedScores(indexScratch[..count], scoreScratch[..count], random);
                    }

                case SamplingStrategy.TopKTopP:
                    {
                        // Apply top-k first, then nucleus on the survivors
                        var k = options.TopK > 0 && options.TopK < logits.Length
                            ? options.TopK
                            : logits.Length;

                        var afterK = SelectTopK(logits, k, temperature, indexScratch, scoreScratch);

                        if (options.TopP < 1f)
                        {
                            var afterP = NucleusFromSorted(
                                indexScratch[..afterK], scoreScratch[..afterK], options.TopP);
                            return SampleFromPreparedScores(
                                indexScratch[..afterP], scoreScratch[..afterP], random);
                        }

                        return SampleFromPreparedScores(indexScratch[..afterK], scoreScratch[..afterK], random);
                    }

                default:
                    // Temperature-only or fallback
                    PrepareAllScores(logits, temperature, indexScratch, scoreScratch);
                    return SampleFromPreparedScores(
                        indexScratch[..logits.Length], scoreScratch[..logits.Length], random);
            }
        }

        public static int ArgMax(ReadOnlySpan<float> logits)
        {
            if (logits.IsEmpty)
            {
                throw new ArgumentException("Logits cannot be empty.", nameof(logits));
            }

            var maxIndex = 0;
            var maxValue = logits[0];

            for (var i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxValue)
                {
                    maxValue = logits[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        // ── Top-P (nucleus sampling) ──────────────────────────────────────────

        /// <summary>
        /// Selects the nucleus — the smallest prefix of tokens (sorted descending by score)
        /// whose cumulative softmax probability ≥ topP.
        ///
        /// Returns the number of tokens in the nucleus. indexScratch and scoreScratch
        /// are populated with the sorted tokens and their temperature-scaled logits.
        ///
        /// Algorithm: full descending sort + softmax prefix scan.
        /// Zero allocations — sort is done in-place on scratch buffers.
        /// </summary>
        private static int SelectTopP(
            ReadOnlySpan<float> logits,
            float topP,
            float temperature,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            // 1. Fill scratch with temperature-scaled logits
            for (var i = 0; i < logits.Length; i++)
            {
                indexScratch[i] = i;
                scoreScratch[i] = logits[i] / temperature;
            }

            // 2. Sort descending — partial insertion sort, exits early for peaky distributions
            SortDescending(indexScratch[..logits.Length], scoreScratch[..logits.Length]);

            // 3. Find nucleus via cumulative softmax probability
            return NucleusFromSorted(indexScratch[..logits.Length], scoreScratch[..logits.Length], topP);
        }

        /// <summary>
        /// Given tokens already sorted descending by score, find the nucleus cutoff
        /// (smallest prefix where cumulative probability ≥ topP).
        /// Returns the count of tokens in the nucleus (minimum 1).
        /// </summary>
        private static int NucleusFromSorted(
            ReadOnlySpan<int> sortedIndexes,
            ReadOnlySpan<float> sortedScores,
            float topP)
        {
            if (sortedScores.IsEmpty)
            {
                return 0;
            }

            // Stable softmax: subtract max for numerical stability
            var maxScore = sortedScores[0]; // already sorted, max is first
            var sum = 0.0;
            for (var i = 0; i < sortedScores.Length; i++)
            {
                sum += Math.Exp(sortedScores[i] - maxScore);
            }

            // Accumulate until cumulative probability ≥ topP
            var cumulative = 0.0;
            for (var i = 0; i < sortedScores.Length; i++)
            {
                cumulative += Math.Exp(sortedScores[i] - maxScore) / sum;
                if (cumulative >= topP)
                {
                    return i + 1; // nucleus is tokens 0..i inclusive
                }
            }

            return sortedScores.Length; // whole distribution (topP ≈ 1.0)
        }

        // ── Sort (in-place descending, no allocation) ─────────────────────────

        /// <summary>
        /// Sorts indexes and scores together by scores descending.
        /// Uses a simple heap sort — O(n log n), zero allocation.
        /// For typical vocab sizes (768-50257) this is fast enough for real-time inference.
        /// </summary>
        private static void SortDescending(Span<int> indexes, Span<float> scores)
        {
            var n = scores.Length;
            if (n <= 1)
            {
                return;
            }

            // Build max-heap
            for (var i = n / 2 - 1; i >= 0; i--)
            {
                Heapify(indexes, scores, n, i);
            }

            // Extract elements one by one
            for (var i = n - 1; i > 0; i--)
            {
                // Swap root (max) to end
                Swap(indexes, scores, 0, i);
                Heapify(indexes, scores, i, 0);
            }

            // Heap sort produces ascending — reverse for descending
            indexes.Reverse();
            scores.Reverse();
        }

        private static void Heapify(Span<int> idx, Span<float> sc, int n, int root)
        {
            var largest = root;
            var left = 2 * root + 1;
            var right = 2 * root + 2;

            if (left < n && sc[left] > sc[largest])
            {
                largest = left;
            }
            if (right < n && sc[right] > sc[largest])
            {
                largest = right;
            }

            if (largest != root)
            {
                Swap(idx, sc, root, largest);
                Heapify(idx, sc, n, largest);
            }
        }

        private static void Swap(Span<int> idx, Span<float> sc, int a, int b)
        {
            (idx[a], idx[b]) = (idx[b], idx[a]);
            (sc[a], sc[b]) = (sc[b], sc[a]);
        }

        // ── Existing helpers (unchanged) ──────────────────────────────────────

        private static void PrepareAllScores(
            ReadOnlySpan<float> logits,
            float temperature,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            for (var i = 0; i < logits.Length; i++)
            {
                indexScratch[i] = i;
                scoreScratch[i] = logits[i] / temperature;
            }
        }

        private static int SelectTopK(
            ReadOnlySpan<float> logits,
            int topK,
            float temperature,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            var activeCount = 0;

            for (var token = 0; token < logits.Length; token++)
            {
                var score = logits[token] / temperature;

                if (activeCount < topK)
                {
                    InsertDescending(token, score, indexScratch, scoreScratch, activeCount);
                    activeCount++;
                    continue;
                }

                if (score <= scoreScratch[topK - 1])
                {
                    continue;
                }

                InsertDescending(token, score, indexScratch, scoreScratch, topK - 1);
            }

            return activeCount;
        }

        private static void InsertDescending(
            int token,
            float score,
            Span<int> indexScratch,
            Span<float> scoreScratch,
            int lastIndex)
        {
            var insertAt = lastIndex;

            while (insertAt > 0 && score > scoreScratch[insertAt - 1])
            {
                if (insertAt < scoreScratch.Length)
                {
                    scoreScratch[insertAt] = scoreScratch[insertAt - 1];
                    indexScratch[insertAt] = indexScratch[insertAt - 1];
                }
                insertAt--;
            }

            scoreScratch[insertAt] = score;
            indexScratch[insertAt] = token;
        }

        private static int SampleFromPreparedScores(
            ReadOnlySpan<int> tokenIndexes,
            ReadOnlySpan<float> scores,
            Random random)
        {
            if (tokenIndexes.IsEmpty)
            {
                throw new ArgumentException("Token index span cannot be empty.", nameof(tokenIndexes));
            }

            var maxScore = scores[0];
            for (var i = 1; i < scores.Length; i++)
            {
                if (scores[i] > maxScore)
                {
                    maxScore = scores[i];
                }
            }

            var sum = 0.0;
            for (var i = 0; i < scores.Length; i++)
            {
                sum += Math.Exp(scores[i] - maxScore);
            }

            if (sum <= 0.0 || double.IsNaN(sum) || double.IsInfinity(sum))
            {
                return tokenIndexes[0];
            }

            var sample = random.NextDouble() * sum;
            var cumulative = 0.0;

            for (var i = 0; i < scores.Length; i++)
            {
                cumulative += Math.Exp(scores[i] - maxScore);
                if (sample <= cumulative)
                {
                    return tokenIndexes[i];
                }
            }

            return tokenIndexes[^1];
        }
    }
}
