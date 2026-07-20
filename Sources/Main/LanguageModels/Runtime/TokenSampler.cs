// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Maths;

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

        // Nucleus / typical sets are tiny in practice, so top-p / typical-p partial-sort just the top candidates
        // (size-k min-heap, O(V·log k)) instead of the whole vocabulary (O(V·log V)) — a ~10× cut on a 150k
        // vocab. If the surviving set is unexpectedly larger than this cap, they fall back to a full sort so the
        // result is always exact. Internal (not const) so tests can shrink it to exercise the fallback path.
        internal static int NucleusPartialCap = 1024;

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

            var count = SelectSurvivors(logits, in options, temperature, indexScratch, scoreScratch);
            return SampleFromPreparedScores(
                indexScratch[..count], scoreScratch[..count], temperature, random);
        }

        /// <summary>
        /// Selects the survivor token set for <paramref name="options"/> (the strategy's top-k / top-p /
        /// min-p filter) into <paramref name="indexScratch"/> (token ids) + <paramref name="scoreScratch"/>
        /// (their raw logits), returning the survivor count. The shared core of <see cref="Sample"/> and
        /// <see cref="ComputeProbabilities"/> — so the speculative verifier scores against EXACTLY the
        /// distribution the sampler would draw from. Assumes a non-greedy strategy (caller handles greedy).
        /// </summary>
        private static int SelectSurvivors(
            ReadOnlySpan<float> logits, in SamplingOptions options, float temperature,
            Span<int> indexScratch, Span<float> scoreScratch)
        {
            switch (options.Strategy)
            {
                case SamplingStrategy.TopK when options.TopK > 0 && options.TopK < logits.Length:
                    return SelectTopK(logits, options.TopK, indexScratch, scoreScratch);

                case SamplingStrategy.TopP when options.TopP < 1f:
                    return SelectTopP(logits, options.TopP, temperature, indexScratch, scoreScratch);

                case SamplingStrategy.TopKTopP:
                    {
                        var k = options.TopK > 0 && options.TopK < logits.Length ? options.TopK : logits.Length;
                        var afterK = SelectTopK(logits, k, indexScratch, scoreScratch);
                        return options.TopP < 1f
                            ? NucleusFromSorted(scoreScratch[..afterK], options.TopP, temperature)
                            : afterK;
                    }

                case SamplingStrategy.MinP when options.MinP > 0f:
                    return SelectMinP(logits, options.MinP, temperature, indexScratch, scoreScratch);

                case SamplingStrategy.TopNSigma when options.NSigma > 0f:
                    return SelectTopNSigma(logits, options.NSigma, indexScratch, scoreScratch);

                case SamplingStrategy.TypicalP when options.TypicalP < 1f:
                    return SelectTypicalP(logits, options.TypicalP, temperature, indexScratch, scoreScratch);

                default:
                    PrepareAllScores(logits, indexScratch, scoreScratch);
                    return logits.Length;
            }
        }

        /// <summary>
        /// Writes the full-vocabulary probability distribution the sampler would draw from for
        /// <paramref name="options"/> into <paramref name="probabilities"/> (survivor tokens get their
        /// temperature-softmax probability, all others 0; greedy → a point mass on the argmax). Used by
        /// sampling-correct speculative decoding to compute the target probability of a draft token and to
        /// resample the residual. Same survivor selection + softmax as <see cref="Sample"/>.
        /// </summary>
        public static void ComputeProbabilities(
            ReadOnlySpan<float> logits,
            in SamplingOptions options,
            Span<int> indexScratch,
            Span<float> scoreScratch,
            Span<float> probabilities)
        {
            if (logits.IsEmpty)
            {
                throw new ArgumentException("Logits cannot be empty.", nameof(logits));
            }
            if (probabilities.Length < logits.Length)
            {
                throw new ArgumentException("Probabilities span is smaller than the logits span.", nameof(probabilities));
            }

            probabilities.Slice(0, logits.Length).Clear();

            if (options.Strategy == SamplingStrategy.Greedy || options.Temperature <= 0f)
            {
                probabilities[ArgMax(logits)] = 1f;
                return;
            }

            var temperature = MathF.Max(options.Temperature, MinimumTemperature);
            var count = SelectSurvivors(logits, in options, temperature, indexScratch, scoreScratch);

            var maxScore = scoreScratch[0];
            for (var i = 1; i < count; i++)
            {
                if (scoreScratch[i] > maxScore)
                {
                    maxScore = scoreScratch[i];
                }
            }

            var inverseTemperature = 1.0 / temperature;
            var sum = 0.0;
            for (var i = 0; i < count; i++)
            {
                var w = Math.Exp((scoreScratch[i] - maxScore) * inverseTemperature);
                scoreScratch[i] = (float)w;
                sum += w;
            }

            if (sum <= 0.0 || double.IsNaN(sum) || double.IsInfinity(sum))
            {
                probabilities[indexScratch[0]] = 1f;
                return;
            }

            for (var i = 0; i < count; i++)
            {
                probabilities[indexScratch[i]] = (float)(scoreScratch[i] / sum);
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

            return MathUtils.ArgMax(logits);
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
                if (logits[i] > maxLogit)
                {
                    maxLogit = logits[i];
                }
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

        // Top-nσ: keep tokens with logit ≥ max − n·σ (σ = std-dev of the finite logits). Acts on raw logits,
        // so it needs no temperature and no sort — a single-threshold pass like Min-P.
        private static int SelectTopNSigma(
            ReadOnlySpan<float> logits, float nSigma, Span<int> indexScratch, Span<float> scoreScratch)
        {
            var count = 0;
            var max = float.NegativeInfinity;
            var sum = 0.0;
            for (var i = 0; i < logits.Length; i++)
            {
                var v = logits[i];
                if (float.IsNegativeInfinity(v))
                {
                    continue;
                }
                count++;
                sum += v;
                if (v > max)
                {
                    max = v;
                }
            }
            if (count == 0)
            {
                return FallbackToArgMax(logits, indexScratch, scoreScratch);
            }

            var mean = sum / count;
            var varSum = 0.0;
            for (var i = 0; i < logits.Length; i++)
            {
                var v = logits[i];
                if (float.IsNegativeInfinity(v))
                {
                    continue;
                }
                var d = v - mean;
                varSum += d * d;
            }
            var threshold = max - (nSigma * (float)Math.Sqrt(varSum / count));

            var kept = 0;
            for (var token = 0; token < logits.Length; token++)
            {
                if (logits[token] >= threshold)
                {
                    indexScratch[kept] = token;
                    scoreScratch[kept] = logits[token];
                    kept++;
                }
            }
            return kept == 0 ? FallbackToArgMax(logits, indexScratch, scoreScratch) : kept;
        }

        // Locally typical sampling: keep the tokens whose surprise (−ln p) is closest to the entropy until
        // their cumulative probability ≥ p. Sorts by deviation (via the existing heap sort on a negated key),
        // then restores the survivors' logits from the original span so the terminal draw softmaxes them.
        private static int SelectTypicalP(
            ReadOnlySpan<float> logits, float typicalP, float temperature,
            Span<int> indexScratch, Span<float> scoreScratch)
        {
            var n = logits.Length;

            var max = float.NegativeInfinity;
            for (var i = 0; i < n; i++)
            {
                if (logits[i] > max)
                {
                    max = logits[i];
                }
            }
            if (float.IsNegativeInfinity(max))
            {
                return FallbackToArgMax(logits, indexScratch, scoreScratch);
            }

            var inverseTemperature = 1.0 / temperature;
            var sum = 0.0;
            for (var i = 0; i < n; i++)
            {
                if (!float.IsNegativeInfinity(logits[i]))
                {
                    sum += Math.Exp((logits[i] - max) * inverseTemperature);
                }
            }
            if (sum <= 0.0)
            {
                return FallbackToArgMax(logits, indexScratch, scoreScratch);
            }

            var entropy = 0.0;
            for (var i = 0; i < n; i++)
            {
                if (float.IsNegativeInfinity(logits[i]))
                {
                    continue;
                }
                var p = Math.Exp((logits[i] - max) * inverseTemperature) / sum;
                if (p > 0.0)
                {
                    entropy -= p * Math.Log(p);
                }
            }

            // Sort key = −deviation, so a descending sort orders tokens by ascending typicality deviation.
            for (var i = 0; i < n; i++)
            {
                indexScratch[i] = i;
                if (float.IsNegativeInfinity(logits[i]))
                {
                    scoreScratch[i] = float.NegativeInfinity;
                    continue;
                }
                var p = Math.Exp((logits[i] - max) * inverseTemperature) / sum;
                var dev = Math.Abs(-Math.Log(p) - entropy);
                scoreScratch[i] = (float)(-dev);
            }

            var cap = Math.Min(n, NucleusPartialCap);
            PartialSortDescendingInPlace(indexScratch[..n], scoreScratch[..n], n, cap);

            var cumulative = 0.0;
            var count = 0;
            var reached = false;
            for (var j = 0; j < cap; j++)
            {
                var token = indexScratch[j];
                if (float.IsNegativeInfinity(logits[token]))
                {
                    break; // masked tokens sort last; none of the real tail is beyond here
                }
                count++;
                cumulative += Math.Exp((logits[token] - max) * inverseTemperature) / sum;
                if (cumulative >= typicalP)
                {
                    reached = true;
                    break;
                }
            }
            if (count == 0)
            {
                return FallbackToArgMax(logits, indexScratch, scoreScratch);
            }

            // Kept set is complete when we hit the target, saw the whole vocab, or ran out of finite tokens
            // within the cap; otherwise the typical set spills past the cap → fall back to a full sort (rare).
            if (reached || cap == n || count < cap)
            {
                for (var j = 0; j < count; j++)
                {
                    scoreScratch[j] = logits[indexScratch[j]];
                }
                return count;
            }
            return SelectTypicalPFull(logits, typicalP, temperature, entropy, max, sum, inverseTemperature, indexScratch, scoreScratch);
        }

        private static int SelectTypicalPFull(
            ReadOnlySpan<float> logits, float typicalP, float temperature, double entropy, float max, double sum,
            double inverseTemperature, Span<int> indexScratch, Span<float> scoreScratch)
        {
            var n = logits.Length;
            for (var i = 0; i < n; i++)
            {
                indexScratch[i] = i;
                if (float.IsNegativeInfinity(logits[i]))
                {
                    scoreScratch[i] = float.NegativeInfinity;
                    continue;
                }
                var p = Math.Exp((logits[i] - max) * inverseTemperature) / sum;
                scoreScratch[i] = (float)(-Math.Abs(-Math.Log(p) - entropy));
            }
            SortDescending(indexScratch[..n], scoreScratch[..n]);

            var cumulative = 0.0;
            var count = 0;
            for (var j = 0; j < n; j++)
            {
                var token = indexScratch[j];
                if (float.IsNegativeInfinity(logits[token]))
                {
                    break;
                }
                count++;
                cumulative += Math.Exp((logits[token] - max) * inverseTemperature) / sum;
                if (cumulative >= typicalP)
                {
                    break;
                }
            }
            if (count == 0)
            {
                return FallbackToArgMax(logits, indexScratch, scoreScratch);
            }
            for (var j = 0; j < count; j++)
            {
                scoreScratch[j] = logits[indexScratch[j]];
            }
            return count;
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
            var n = logits.Length;

            var max = float.NegativeInfinity;
            for (var i = 0; i < n; i++)
            {
                if (logits[i] > max)
                {
                    max = logits[i];
                }
            }
            if (float.IsNegativeInfinity(max))
            {
                return FallbackToArgMax(logits, indexScratch, scoreScratch);
            }

            var inverseTemperature = 1.0 / temperature;
            var sum = 0.0;
            for (var i = 0; i < n; i++)
            {
                if (!float.IsNegativeInfinity(logits[i]))
                {
                    sum += Math.Exp((logits[i] - max) * inverseTemperature);
                }
            }
            if (sum <= 0.0)
            {
                return FallbackToArgMax(logits, indexScratch, scoreScratch);
            }

            PrepareAllScores(logits, indexScratch, scoreScratch);
            var cap = Math.Min(n, NucleusPartialCap);
            PartialSortDescendingInPlace(indexScratch[..n], scoreScratch[..n], n, cap);

            // The top `cap` logits are now sorted descending; walk their cumulative probability.
            var cumulative = 0.0;
            for (var j = 0; j < cap; j++)
            {
                cumulative += Math.Exp((scoreScratch[j] - max) * inverseTemperature) / sum;
                if (cumulative >= topP)
                {
                    return j + 1;
                }
            }

            if (cap == n)
            {
                return n; // saw the whole vocab; all survive
            }
            return SelectTopPFull(logits, topP, temperature, indexScratch, scoreScratch); // nucleus exceeds cap (rare)
        }

        private static int SelectTopPFull(
            ReadOnlySpan<float> logits, float topP, float temperature, Span<int> indexScratch, Span<float> scoreScratch)
        {
            PrepareAllScores(logits, indexScratch, scoreScratch);
            SortDescending(indexScratch[..logits.Length], scoreScratch[..logits.Length]);
            return NucleusFromSorted(scoreScratch[..logits.Length], topP, temperature);
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

        // Arranges the k highest-scored elements into [0, k) in DESCENDING score order (elements beyond k are
        // left unspecified). O(count·log k) via a size-k min-heap — far cheaper than a full sort when k ≪ count,
        // which is the normal case for a nucleus / typical set over a large vocabulary.
        private static void PartialSortDescendingInPlace(Span<int> indexes, Span<float> scores, int count, int k)
        {
            if (k >= count)
            {
                SortDescending(indexes[..count], scores[..count]);
                return;
            }

            // Build a min-heap over the first k, then let any larger element beyond k evict the current minimum.
            for (var i = k / 2 - 1; i >= 0; i--)
            {
                SiftDownMin(indexes, scores, k, i);
            }
            for (var i = k; i < count; i++)
            {
                if (scores[i] > scores[0])
                {
                    (scores[0], scores[i]) = (scores[i], scores[0]);
                    (indexes[0], indexes[i]) = (indexes[i], indexes[0]);
                    SiftDownMin(indexes, scores, k, 0);
                }
            }
            SortDescending(indexes[..k], scores[..k]);
        }

        private static void SiftDownMin(Span<int> indexes, Span<float> scores, int length, int root)
        {
            // BOUND: <= log2(length). Each iteration moves `root` to a child (2*root+1 or +2) or returns,
            // so the loop descends one heap level per pass and cannot revisit a node.
#pragma warning disable OVERFIT023
            while (true)
#pragma warning restore OVERFIT023
            {
                var smallest = root;
                var left = 2 * root + 1;
                var right = 2 * root + 2;

                if (left < length && scores[left] < scores[smallest])
                {
                    smallest = left;
                }
                if (right < length && scores[right] < scores[smallest])
                {
                    smallest = right;
                }
                if (smallest == root)
                {
                    return;
                }

                Swap(indexes, scores, root, smallest);
                root = smallest;
            }
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

            // Sift-down: `largest` is always a CHILD of `root` (2*root+1 or +2), so each call descends one
            // heap level. Depth is therefore <= log2(length) — ~18 for a 150k vocabulary, not input-shaped.
#pragma warning disable OVERFIT022 // Bounded: descends one heap level per call, depth <= log2(length).
            Heapify(
                indexes,
                scores,
                length,
                largest);
#pragma warning restore OVERFIT022
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
