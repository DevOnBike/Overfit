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
    /// Greedy sampling is the hot path for parity/validation and does not require
    /// Random or scratch buffers.
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
                throw new ArgumentException(
                    "Logits cannot be empty.",
                    nameof(logits));
            }

            if (options.Strategy == SamplingStrategy.Greedy ||
                options.Temperature <= 0f)
            {
                return ArgMax(logits);
            }

            if (random is null)
            {
                throw new ArgumentNullException(nameof(random));
            }

            if (indexScratch.Length < logits.Length)
            {
                throw new ArgumentException(
                    "Index scratch span is smaller than the logits span.",
                    nameof(indexScratch));
            }

            if (scoreScratch.Length < logits.Length)
            {
                throw new ArgumentException(
                    "Score scratch span is smaller than the logits span.",
                    nameof(scoreScratch));
            }

            var temperature = MathF.Max(
                options.Temperature,
                MinimumTemperature);

            switch (options.Strategy)
            {
                case SamplingStrategy.TopK
                    when options.TopK > 0 && options.TopK < logits.Length:
                    {
                        var count = SelectTopK(
                            logits,
                            options.TopK,
                            indexScratch,
                            scoreScratch);

                        return SampleFromPreparedScores(
                            indexScratch[..count],
                            scoreScratch[..count],
                            temperature,
                            random);
                    }

                case SamplingStrategy.TopP
                    when options.TopP < 1f:
                    {
                        var count = SelectTopP(
                            logits,
                            options.TopP,
                            temperature,
                            indexScratch,
                            scoreScratch);

                        return SampleFromPreparedScores(
                            indexScratch[..count],
                            scoreScratch[..count],
                            temperature,
                            random);
                    }

                case SamplingStrategy.TopKTopP:
                    {
                        var k =
                            options.TopK > 0 && options.TopK < logits.Length
                                ? options.TopK
                                : logits.Length;

                        var afterK = SelectTopK(
                            logits,
                            k,
                            indexScratch,
                            scoreScratch);

                        if (options.TopP < 1f)
                        {
                            var afterP = NucleusFromSorted(
                                scoreScratch[..afterK],
                                options.TopP,
                                temperature);

                            return SampleFromPreparedScores(
                                indexScratch[..afterP],
                                scoreScratch[..afterP],
                                temperature,
                                random);
                        }

                        return SampleFromPreparedScores(
                            indexScratch[..afterK],
                            scoreScratch[..afterK],
                            temperature,
                            random);
                    }

                case SamplingStrategy.MinP
                    when options.MinP > 0f:
                    {
                        var count = SelectMinP(
                            logits,
                            options.MinP,
                            temperature,
                            indexScratch,
                            scoreScratch);

                        return SampleFromPreparedScores(
                            indexScratch[..count],
                            scoreScratch[..count],
                            temperature,
                            random);
                    }

                default:
                    {
                        PrepareAllScores(
                            logits,
                            indexScratch,
                            scoreScratch);

                        return SampleFromPreparedScores(
                            indexScratch[..logits.Length],
                            scoreScratch[..logits.Length],
                            temperature,
                            random);
                    }
            }
        }

        public static int ArgMax(
            ReadOnlySpan<float> logits)
        {
            if (logits.IsEmpty)
            {
                throw new ArgumentException(
                    "Logits cannot be empty.",
                    nameof(logits));
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

        /// <summary>
        /// Applies HuggingFace-style repetition penalty in-place.
        /// For each token in <paramref name="recentTokens"/>:
        ///   if logits[token] &lt; 0: logits[token] *= penalty
        ///   else:                   logits[token] /= penalty
        /// No-op when penalty &lt;= 1.0 or recentTokens is empty.
        /// Duplicate tokens in recentTokens are applied multiple times intentionally —
        /// repeated tokens get penalized more (matches transformers reference impl).
        /// </summary>
        public static void ApplyRepetitionPenalty(
            Span<float> logits,
            ReadOnlySpan<int> recentTokens,
            float penalty)
        {
            if (penalty <= 1.0f || recentTokens.IsEmpty)
            {
                return;
            }

            for (var i = 0; i < recentTokens.Length; i++)
            {
                var token = recentTokens[i];

                if (token < 0 || token >= logits.Length)
                {
                    continue;
                }

                var logit = logits[token];

                logits[token] = logit < 0f ? logit * penalty : logit / penalty;
            }
        }

        private static void PrepareAllScores(
            ReadOnlySpan<float> logits,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            for (var i = 0; i < logits.Length; i++)
            {
                indexScratch[i] = i;
                scoreScratch[i] = logits[i];
            }
        }

        // Min-P filter: keep tokens with P(token) ≥ minP × P(top). In logit space that is
        // (logit − maxLogit)/T ≥ ln(minP), i.e. logit ≥ maxLogit + T·ln(minP). No sort needed.
        private static int SelectMinP(
            ReadOnlySpan<float> logits,
            float minP,
            float temperature,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            var maxLogit = logits[0];
            for (var i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxLogit) { maxLogit = logits[i]; }
            }

            var threshold = maxLogit + (temperature * MathF.Log(Math.Clamp(minP, 1e-6f, 1f)));
            var count = 0;
            for (var token = 0; token < logits.Length; token++)
            {
                if (logits[token] >= threshold)
                {
                    indexScratch[count] = token;
                    scoreScratch[count] = logits[token];
                    count++;
                }
            }
            return count == 0 ? FallbackToArgMax(logits, indexScratch, scoreScratch) : count;
        }

        private static int FallbackToArgMax(ReadOnlySpan<float> logits, Span<int> indexScratch, Span<float> scoreScratch)
        {
            var best = ArgMax(logits);
            indexScratch[0] = best;
            scoreScratch[0] = logits[best];
            return 1;
        }

        private static int SelectTopK(
            ReadOnlySpan<float> logits,
            int topK,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            var activeCount = 0;

            for (var token = 0; token < logits.Length; token++)
            {
                var score = logits[token];

                if (activeCount < topK)
                {
                    InsertDescending(
                        token,
                        score,
                        indexScratch,
                        scoreScratch,
                        activeCount);

                    activeCount++;
                    continue;
                }

                if (score <= scoreScratch[topK - 1])
                {
                    continue;
                }

                InsertDescending(
                    token,
                    score,
                    indexScratch,
                    scoreScratch,
                    topK - 1);
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

        private static int SelectTopP(
            ReadOnlySpan<float> logits,
            float topP,
            float temperature,
            Span<int> indexScratch,
            Span<float> scoreScratch)
        {
            PrepareAllScores(
                logits,
                indexScratch,
                scoreScratch);

            SortDescending(
                indexScratch[..logits.Length],
                scoreScratch[..logits.Length]);

            return NucleusFromSorted(
                scoreScratch[..logits.Length],
                topP,
                temperature);
        }

        private static int NucleusFromSorted(
            ReadOnlySpan<float> sortedScores,
            float topP,
            float temperature)
        {
            if (sortedScores.IsEmpty)
            {
                return 0;
            }

            var maxScore = sortedScores[0];
            var inverseTemperature = 1.0 / temperature;

            var sum = 0.0;

            for (var i = 0; i < sortedScores.Length; i++)
            {
                sum += Math.Exp((sortedScores[i] - maxScore) * inverseTemperature);
            }

            if (sum <= 0.0 ||
                double.IsNaN(sum) ||
                double.IsInfinity(sum))
            {
                return 1;
            }

            var cumulative = 0.0;

            for (var i = 0; i < sortedScores.Length; i++)
            {
                cumulative +=
                    Math.Exp(
                        (sortedScores[i] - maxScore) * inverseTemperature) /
                    sum;

                if (cumulative >= topP)
                {
                    return i + 1;
                }
            }

            return sortedScores.Length;
        }

        private static int SampleFromPreparedScores(
            Span<int> tokenIndexes,
            Span<float> scores,
            float temperature,
            Random random)
        {
            if (tokenIndexes.IsEmpty)
            {
                throw new ArgumentException(
                    "Token index span cannot be empty.",
                    nameof(tokenIndexes));
            }

            var maxScore = scores[0];

            for (var i = 1; i < scores.Length; i++)
            {
                if (scores[i] > maxScore)
                {
                    maxScore = scores[i];
                }
            }

            var inverseTemperature = 1.0 / temperature;
            var sum = 0.0;

            for (var i = 0; i < scores.Length; i++)
            {
                var weight = Math.Exp(
                    (scores[i] - maxScore) * inverseTemperature);

                scores[i] = (float)weight;
                sum += weight;
            }

            if (sum <= 0.0 ||
                double.IsNaN(sum) ||
                double.IsInfinity(sum))
            {
                return tokenIndexes[0];
            }

            var sample = random.NextDouble() * sum;
            var cumulative = 0.0;

            for (var i = 0; i < scores.Length; i++)
            {
                cumulative += scores[i];

                if (sample <= cumulative)
                {
                    return tokenIndexes[i];
                }
            }

            return tokenIndexes[^1];
        }

        private static void SortDescending(
            Span<int> indexes,
            Span<float> scores)
        {
            var n = scores.Length;

            if (n <= 1)
            {
                return;
            }

            for (var i = n / 2 - 1; i >= 0; i--)
            {
                Heapify(
                    indexes,
                    scores,
                    n,
                    i);
            }

            for (var i = n - 1; i > 0; i--)
            {
                Swap(
                    indexes,
                    scores,
                    0,
                    i);

                Heapify(
                    indexes,
                    scores,
                    i,
                    0);
            }

            indexes.Reverse();
            scores.Reverse();
        }

        private static void Heapify(
            Span<int> indexes,
            Span<float> scores,
            int length,
            int root)
        {
            var largest = root;
            var left = 2 * root + 1;
            var right = 2 * root + 2;

            if (left < length && scores[left] > scores[largest])
            {
                largest = left;
            }

            if (right < length && scores[right] > scores[largest])
            {
                largest = right;
            }

            if (largest == root)
            {
                return;
            }

            Swap(
                indexes,
                scores,
                root,
                largest);

            Heapify(
                indexes,
                scores,
                length,
                largest);
        }

        private static void Swap(
            Span<int> indexes,
            Span<float> scores,
            int a,
            int b)
        {
            (indexes[a], indexes[b]) = (indexes[b], indexes[a]);
            (scores[a], scores[b]) = (scores[b], scores[a]);
        }
    }
}
