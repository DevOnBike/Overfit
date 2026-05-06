// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Cached single-head attention decode block.
    ///
    /// This is the first composed decode building block:
    ///
    /// 1. hidden[token] -> Q
    /// 2. hidden[token] -> K
    /// 3. hidden[token] -> V
    /// 4. write K/V into KeyValueCache at the current position
    /// 5. cached attention over K/V positions [0..position]
    /// 6. attention output -> output projection
    ///
    /// Scope:
    /// - batch = 1
    /// - one layer
    /// - one head
    /// - FP32
    /// - caller-owned output
    /// - cache lifetime controlled by caller
    ///
    /// Important:
    /// The caller must advance the cache length before calling Decode for a
    /// position that should be visible to attention:
    ///
    /// cache.Advance();
    /// decoder.Decode(..., position: cache.CurrentLength - 1, ...);
    ///
    /// This keeps cache length controlled at a higher level where all heads/layers
    /// can be coordinated. The single-head decoder only writes K/V and reads the
    /// visible cache range.
    /// </summary>
    public sealed class CachedSingleHeadAttention
    {
        private readonly float[] _query;
        private readonly float[] _key;
        private readonly float[] _value;
        private readonly float[] _attentionOutput;
        private readonly float[] _scoreScratch;
        private readonly float _scale;

        public CachedSingleHeadAttention(
            int dModel,
            int headDimension,
            int maxSequenceLength)
        {
            if (dModel <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(dModel));
            }

            if (headDimension <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(headDimension));
            }

            if (maxSequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
            }

            DModel = dModel;
            HeadDimension = headDimension;
            MaxSequenceLength = maxSequenceLength;

            _query = new float[headDimension];
            _key = new float[headDimension];
            _value = new float[headDimension];
            _attentionOutput = new float[headDimension];
            _scoreScratch = new float[maxSequenceLength];
            _scale = 1f / MathF.Sqrt(headDimension);
        }

        public int DModel { get; }

        public int HeadDimension { get; }

        public int MaxSequenceLength { get; }

        public void Decode(
            ReadOnlySpan<float> hidden,
            ReadOnlySpan<float> wq,
            ReadOnlySpan<float> wk,
            ReadOnlySpan<float> wv,
            ReadOnlySpan<float> bq,
            ReadOnlySpan<float> bk,
            ReadOnlySpan<float> bv,
            ReadOnlySpan<float> wo,
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position,
            Span<float> output)
        {
            SingleTokenProjectionKernel.Project(
                hidden,
                wq,
                bq,
                _query,
                DModel,
                HeadDimension);

            SingleTokenProjectionKernel.Project(
                hidden,
                wk,
                bk,
                _key,
                DModel,
                HeadDimension);

            SingleTokenProjectionKernel.Project(
                hidden,
                wv,
                bv,
                _value,
                DModel,
                HeadDimension);

            _key
                .AsSpan()
                .CopyTo(cache.GetKeyWriteSpan(layerIndex, headIndex, position));

            _value
                .AsSpan()
                .CopyTo(cache.GetValueWriteSpan(layerIndex, headIndex, position));

            var sequenceLength = position + 1;

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

            CachedAttentionKernel.ComputeSingleHead(
                _query,
                keys,
                values,
                _attentionOutput,
                _scoreScratch,
                sequenceLength,
                HeadDimension,
                _scale);

            SingleTokenProjectionKernel.Project(
                _attentionOutput,
                wo,
                ReadOnlySpan<float>.Empty,
                output,
                HeadDimension,
                DModel);
        }


        public void GetLastQuery(Span<float> destination)
        {
            if (destination.Length < HeadDimension)
            {
                throw new ArgumentException("Destination span is smaller than head dimension.", nameof(destination));
            }

            _query.AsSpan().CopyTo(destination);
        }

        public void GetLastKey(Span<float> destination)
        {
            if (destination.Length < HeadDimension)
            {
                throw new ArgumentException("Destination span is smaller than head dimension.", nameof(destination));
            }

            _key.AsSpan().CopyTo(destination);
        }

        public void GetLastValue(Span<float> destination)
        {
            if (destination.Length < HeadDimension)
            {
                throw new ArgumentException("Destination span is smaller than head dimension.", nameof(destination));
            }

            _value.AsSpan().CopyTo(destination);
        }

        public void GetLastAttentionOutput(Span<float> destination)
        {
            if (destination.Length < HeadDimension)
            {
                throw new ArgumentException("Destination span is smaller than head dimension.", nameof(destination));
            }

            _attentionOutput.AsSpan().CopyTo(destination);
        }

    }
}
