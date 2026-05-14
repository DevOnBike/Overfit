// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Single-query cached attention kernel for autoregressive SLM decode.
    ///
    /// This kernel computes one attention head for one newly generated token:
    ///
    /// scores[t] = dot(query, key[t]) * scale
    /// probs = softmax(scores)
    /// output = sum_t(probs[t] * value[t])
    ///
    /// Expected layout:
    ///
    /// query:  [headDim]
    /// keys:   [sequenceLength, headDim]
    /// values: [sequenceLength, headDim]
    /// output: [headDim]
    /// scoreScratch: [sequenceLength]
    ///
    /// This is intentionally small and allocation-free. It does not own buffers,
    /// does not allocate scratch arrays and does not know about layers or heads.
    /// The caller supplies slices from KeyValueCache.
    /// </summary>
    public static class CachedAttentionKernel
    {
        public static void ComputeSingleHead(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int sequenceLength,
            int headDimension,
            float scale)
        {
            ValidateArguments(
                query,
                keys,
                values,
                output,
                scoreScratch,
                sequenceLength,
                headDimension);

            if (sequenceLength == 0)
            {
                output.Slice(0, headDimension).Clear();
                return;
            }

            var maxScore = float.NegativeInfinity;

            for (var t = 0; t < sequenceLength; t++)
            {
                var key = keys.Slice(t * headDimension, headDimension);
                var score = Dot(query, key) * scale;

                scoreScratch[t] = score;

                if (score > maxScore)
                {
                    maxScore = score;
                }
            }

            var sumExp = 0.0f;

            for (var t = 0; t < sequenceLength; t++)
            {
                var exp = MathF.Exp(scoreScratch[t] - maxScore);
                scoreScratch[t] = exp;
                sumExp += exp;
            }

            if (sumExp <= 0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
            {
                output.Slice(0, headDimension).Clear();
                return;
            }

            output.Slice(0, headDimension).Clear();

            var invSum = 1f / sumExp;

            for (var t = 0; t < sequenceLength; t++)
            {
                var probability = scoreScratch[t] * invSum;
                var value = values.Slice(t * headDimension, headDimension);

                for (var d = 0; d < headDimension; d++)
                {
                    output[d] += probability * value[d];
                }
            }
        }

        public static void ComputeSingleHeadFromCache(
            IKeyValueCacheReader cache,
            int layerIndex,
            int headIndex,
            ReadOnlySpan<float> query,
            Span<float> output,
            Span<float> scoreScratch,
            float scale)
        {
            if (cache is null)
            {
                throw new ArgumentNullException(nameof(cache));
            }

            var sequenceLength = cache.CurrentLength;
            var headDimension = cache.HeadDimension;

            var keys = cache.GetKeyReadSpan(
                layerIndex,
                headIndex,
                fromPosition: 0,
                length: sequenceLength);

            var values = cache.GetValueReadSpan(
                layerIndex,
                headIndex,
                fromPosition: 0,
                length: sequenceLength);

            ComputeSingleHead(
                query,
                keys,
                values,
                output,
                scoreScratch,
                sequenceLength,
                headDimension,
                scale);
        }

        private static void ValidateArguments(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int sequenceLength,
            int headDimension)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(sequenceLength);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

            if (query.Length < headDimension)
            {
                throw new ArgumentException("Query span is smaller than headDimension.", nameof(query));
            }

            if (output.Length < headDimension)
            {
                throw new ArgumentException("Output span is smaller than headDimension.", nameof(output));
            }

            if (scoreScratch.Length < sequenceLength)
            {
                throw new ArgumentException("Score scratch span is smaller than sequenceLength.", nameof(scoreScratch));
            }

            var requiredKvLength = sequenceLength * headDimension;

            if (keys.Length < requiredKvLength)
            {
                throw new ArgumentException("Keys span is smaller than sequenceLength * headDimension.", nameof(keys));
            }

            if (values.Length < requiredKvLength)
            {
                throw new ArgumentException("Values span is smaller than sequenceLength * headDimension.", nameof(values));
            }
        }

        private static float Dot(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right)
        {
            var sum = 0f;

            for (var i = 0; i < left.Length; i++)
            {
                sum += left[i] * right[i];
            }

            return sum;
        }
    }

    /// <summary>
    /// Small read-only cache view used by CachedAttentionKernel.
    ///
    /// KeyValueCache implements the same methods through IKeyValueCache already.
    /// This interface exists to keep cached attention decoupled from the full
    /// mutable cache contract and make future specialized cache views possible.
    /// </summary>
    public interface IKeyValueCacheReader
    {
        int CurrentLength { get; }

        int HeadDimension { get; }

        ReadOnlySpan<float> GetKeyReadSpan(
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length);

        ReadOnlySpan<float> GetValueReadSpan(
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length);
    }
}
