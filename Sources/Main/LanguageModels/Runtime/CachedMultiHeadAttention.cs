// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Cached multi-head attention decode block.
    ///
    /// This composes CachedSingleHeadAttention across all heads for one token:
    ///
    /// for each head:
    ///   hidden -> Q/K/V
    ///   write K/V to KeyValueCache[layer, head, position]
    ///   cached attention over positions [0..position]
    ///   per-head output projection -> dModel
    ///
    /// final output:
    ///   outputBias + sum(headProjectedOutput)
    ///
    /// Scope:
    /// - batch = 1,
    /// - one transformer layer,
    /// - FP32,
    /// - caller-owned output buffer,
    /// - caller controls cache length.
    ///
    /// Important:
    /// The caller must call cache.Advance() before decoding the new position.
    /// The cache length must already include the position being decoded.
    /// </summary>
    public sealed class CachedMultiHeadAttention
    {
        private readonly CachedSingleHeadAttention[] _heads;
        private readonly float[] _headOutput;

        public CachedMultiHeadAttention(
            int dModel,
            int headCount,
            int maxSequenceLength)
        {
            if (dModel <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(dModel));
            }

            if (headCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(headCount));
            }

            if (dModel % headCount != 0)
            {
                throw new ArgumentException(
                    $"dModel ({dModel}) must be divisible by headCount ({headCount}).",
                    nameof(dModel));
            }

            if (maxSequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
            }

            DModel = dModel;
            HeadCount = headCount;
            HeadDimension = dModel / headCount;
            MaxSequenceLength = maxSequenceLength;

            _heads = new CachedSingleHeadAttention[headCount];

            for (var h = 0; h < headCount; h++)
            {
                _heads[h] = new CachedSingleHeadAttention(
                    dModel,
                    HeadDimension,
                    maxSequenceLength);
            }

            _headOutput = new float[dModel];
        }

        public int DModel { get; }

        public int HeadCount { get; }

        public int HeadDimension { get; }

        public int MaxSequenceLength { get; }

        public void Decode(
            ReadOnlySpan<float> hidden,
            IReadOnlyList<float[]> wqHeads,
            IReadOnlyList<float[]> wkHeads,
            IReadOnlyList<float[]> wvHeads,
            IReadOnlyList<float[]> woHeads,
            ReadOnlySpan<float> outputBias,
            KeyValueCache cache,
            int layerIndex,
            int position,
            Span<float> output)
        {
            ValidateDecodeArguments(
                hidden,
                wqHeads,
                wkHeads,
                wvHeads,
                woHeads,
                outputBias,
                cache,
                position,
                output);

            if (outputBias.IsEmpty)
            {
                output.Slice(0, DModel).Clear();
            }
            else
            {
                outputBias
                    .Slice(0, DModel)
                    .CopyTo(output);
            }

            for (var h = 0; h < HeadCount; h++)
            {
                _heads[h].DecodeWithoutOutputBias(
                    hidden,
                    wqHeads[h],
                    wkHeads[h],
                    wvHeads[h],
                    woHeads[h],
                    cache,
                    layerIndex,
                    h,
                    position,
                    _headOutput);

                AddInPlace(
                    _headOutput,
                    output,
                    DModel);
            }
        }

        public void DecodeWithoutOutputBias(
            ReadOnlySpan<float> hidden,
            IReadOnlyList<float[]> wqHeads,
            IReadOnlyList<float[]> wkHeads,
            IReadOnlyList<float[]> wvHeads,
            IReadOnlyList<float[]> woHeads,
            KeyValueCache cache,
            int layerIndex,
            int position,
            Span<float> output)
        {
            Decode(
                hidden,
                wqHeads,
                wkHeads,
                wvHeads,
                woHeads,
                ReadOnlySpan<float>.Empty,
                cache,
                layerIndex,
                position,
                output);
        }

        public CachedSingleHeadAttention GetHeadDecoder(int headIndex)
        {
            if ((uint)headIndex >= (uint)HeadCount)
            {
                throw new ArgumentOutOfRangeException(nameof(headIndex));
            }

            return _heads[headIndex];
        }

        private void ValidateDecodeArguments(
            ReadOnlySpan<float> hidden,
            IReadOnlyList<float[]> wqHeads,
            IReadOnlyList<float[]> wkHeads,
            IReadOnlyList<float[]> wvHeads,
            IReadOnlyList<float[]> woHeads,
            ReadOnlySpan<float> outputBias,
            KeyValueCache cache,
            int position,
            Span<float> output)
        {
            if (hidden.Length < DModel)
            {
                throw new ArgumentException("Hidden span is smaller than dModel.", nameof(hidden));
            }

            if (wqHeads is null)
            {
                throw new ArgumentNullException(nameof(wqHeads));
            }

            if (wkHeads is null)
            {
                throw new ArgumentNullException(nameof(wkHeads));
            }

            if (wvHeads is null)
            {
                throw new ArgumentNullException(nameof(wvHeads));
            }

            if (woHeads is null)
            {
                throw new ArgumentNullException(nameof(woHeads));
            }

            if (wqHeads.Count < HeadCount)
            {
                throw new ArgumentException("Wq heads collection is smaller than headCount.", nameof(wqHeads));
            }

            if (wkHeads.Count < HeadCount)
            {
                throw new ArgumentException("Wk heads collection is smaller than headCount.", nameof(wkHeads));
            }

            if (wvHeads.Count < HeadCount)
            {
                throw new ArgumentException("Wv heads collection is smaller than headCount.", nameof(wvHeads));
            }

            if (woHeads.Count < HeadCount)
            {
                throw new ArgumentException("Wo heads collection is smaller than headCount.", nameof(woHeads));
            }

            var qkvWeightsLength = DModel * HeadDimension;
            var outputWeightsLength = HeadDimension * DModel;

            for (var h = 0; h < HeadCount; h++)
            {
                if (wqHeads[h] is null || wqHeads[h].Length < qkvWeightsLength)
                {
                    throw new ArgumentException($"Wq head {h} is smaller than dModel * headDimension.", nameof(wqHeads));
                }

                if (wkHeads[h] is null || wkHeads[h].Length < qkvWeightsLength)
                {
                    throw new ArgumentException($"Wk head {h} is smaller than dModel * headDimension.", nameof(wkHeads));
                }

                if (wvHeads[h] is null || wvHeads[h].Length < qkvWeightsLength)
                {
                    throw new ArgumentException($"Wv head {h} is smaller than dModel * headDimension.", nameof(wvHeads));
                }

                if (woHeads[h] is null || woHeads[h].Length < outputWeightsLength)
                {
                    throw new ArgumentException($"Wo head {h} is smaller than headDimension * dModel.", nameof(woHeads));
                }
            }

            if (!outputBias.IsEmpty && outputBias.Length < DModel)
            {
                throw new ArgumentException("Output bias span is smaller than dModel.", nameof(outputBias));
            }

            if (cache is null)
            {
                throw new ArgumentNullException(nameof(cache));
            }

            if (cache.Shape.HeadCount < HeadCount)
            {
                throw new ArgumentException(
                    $"Cache head count {cache.Shape.HeadCount} is smaller than decoder head count {HeadCount}.",
                    nameof(cache));
            }

            if (cache.Shape.HeadDimension != HeadDimension)
            {
                throw new ArgumentException(
                    $"Cache head dimension {cache.Shape.HeadDimension} does not match decoder head dimension {HeadDimension}.",
                    nameof(cache));
            }

            if (position < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(position));
            }

            if (position >= MaxSequenceLength)
            {
                throw new ArgumentOutOfRangeException(nameof(position));
            }

            if (position >= cache.CurrentLength)
            {
                throw new InvalidOperationException(
                    $"Position {position} is not visible in the cache. CurrentLength={cache.CurrentLength}. Advance the cache before Decode.");
            }

            if (output.Length < DModel)
            {
                throw new ArgumentException("Output span is smaller than dModel.", nameof(output));
            }
        }

        private static void AddInPlace(
            ReadOnlySpan<float> source,
            Span<float> destination,
            int length)
        {
            for (var i = 0; i < length; i++)
            {
                destination[i] += source[i];
            }
        }
    }
}
