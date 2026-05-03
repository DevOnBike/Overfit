// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
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
    /// Current support:
    /// - Greedy
    /// - Temperature sampling
    /// - Top-k sampling
    ///
    /// Top-p / nucleus sampling is intentionally left for a later PR because it
    /// needs a sorted probability prefix. Keeping it out for now avoids hiding
    /// allocations or an expensive full-vocabulary sort in the first runtime shell.
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
                throw new ArgumentException("Index scratch span is smaller than the logits span.", nameof(indexScratch));
            }

            if (scoreScratch.Length < logits.Length)
            {
                throw new ArgumentException("Score scratch span is smaller than the logits span.", nameof(scoreScratch));
            }

            if (options.Strategy == SamplingStrategy.Greedy || options.Temperature <= 0f)
            {
                return ArgMax(logits);
            }

            if (options.Strategy == SamplingStrategy.TopP ||
                (options.Strategy == SamplingStrategy.TopKTopP && options.TopP < 1f))
            {
                throw new NotSupportedException(
                    "Top-p sampling is not implemented in the first SLM runtime skeleton. Use Greedy, Temperature or TopK.");
            }

            var temperature = MathF.Max(options.Temperature, MinimumTemperature);

            if ((options.Strategy == SamplingStrategy.TopK ||
                 options.Strategy == SamplingStrategy.TopKTopP) &&
                options.TopK > 0 &&
                options.TopK < logits.Length)
            {
                var activeCount = SelectTopK(
                    logits,
                    options.TopK,
                    temperature,
                    indexScratch,
                    scoreScratch);

                return SampleFromPreparedScores(
                    indexScratch.Slice(0, activeCount),
                    scoreScratch.Slice(0, activeCount),
                    random);
            }

            PrepareAllScores(
                logits,
                temperature,
                indexScratch,
                scoreScratch);

            return SampleFromPreparedScores(
                indexScratch.Slice(0, logits.Length),
                scoreScratch.Slice(0, logits.Length),
                random);
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
