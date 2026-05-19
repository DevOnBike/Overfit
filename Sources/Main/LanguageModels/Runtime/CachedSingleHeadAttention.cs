// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Rope;

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
        private readonly sbyte[] _q8InputQuants;   // Q8 activation scratch (quantized projection path)
        private readonly float[] _q8InputScales;
        private readonly float _scale;

        public CachedSingleHeadAttention(
            int dModel,
            int headDimension,
            int maxSequenceLength)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);

            DModel = dModel;
            HeadDimension = headDimension;
            MaxSequenceLength = maxSequenceLength;

            _query = new float[headDimension];
            _key = new float[headDimension];
            _value = new float[headDimension];
            _attentionOutput = new float[headDimension];
            _scoreScratch = new float[maxSequenceLength];

            // Q8 activation scratch — sized for the largest projection input
            // (dModel for Q/K/V; headDim ≤ dModel for the output projection).
            _q8InputQuants = new sbyte[dModel];
            _q8InputScales = new float[(dModel + Q8DotKernel.BlockSize - 1) / Q8DotKernel.BlockSize];

            _scale = 1f / MathF.Sqrt(headDimension);
        }

        public int DModel { get; }

        public int HeadDimension { get; }

        public int MaxSequenceLength { get; }

        /// <param name="rope">
        /// Precomputed RoPE table. When non-null, RoPE is applied to Q and K
        /// after projection and before writing K to the cache.
        /// Cached K vectors are stored already rotated — no re-rotation at read time.
        /// </param>
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
            Span<float> output,
            RopeTable? rope = null)
        {
            SingleTokenProjectionKernel.Project(hidden, wq, bq, _query, DModel, HeadDimension);
            SingleTokenProjectionKernel.Project(hidden, wk, bk, _key, DModel, HeadDimension);
            SingleTokenProjectionKernel.Project(hidden, wv, bv, _value, DModel, HeadDimension);

            AttendCore(cache, layerIndex, headIndex, position, rope);

            SingleTokenProjectionKernel.Project(
                _attentionOutput, wo, ReadOnlySpan<float>.Empty, output, HeadDimension, DModel);
        }

        /// <summary>
        /// Q8_0-resident counterpart of <see cref="Decode"/> — the GGUF/Qwen
        /// decode path once attention weights are quantized (step 2.3b-attn).
        /// Same math; the four projections run through <see cref="Q8DotKernel"/>.
        /// </summary>
        public void DecodeQuantized(
            ReadOnlySpan<float> hidden,
            Q8Weight wq,
            Q8Weight wk,
            Q8Weight wv,
            ReadOnlySpan<float> bq,
            ReadOnlySpan<float> bk,
            ReadOnlySpan<float> bv,
            Q8Weight wo,
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position,
            Span<float> output,
            RopeTable? rope = null)
        {
            Q8DotKernel.Project(hidden, wq, bq, _query, _q8InputQuants, _q8InputScales);
            Q8DotKernel.Project(hidden, wk, bk, _key, _q8InputQuants, _q8InputScales);
            Q8DotKernel.Project(hidden, wv, bv, _value, _q8InputQuants, _q8InputScales);

            AttendCore(cache, layerIndex, headIndex, position, rope);

            Q8DotKernel.Project(
                _attentionOutput, wo, ReadOnlySpan<float>.Empty, output, _q8InputQuants, _q8InputScales);
        }

        /// <summary>
        /// Shared decode middle — RoPE on Q/K, K/V cache write, cached attention
        /// into <see cref="_attentionOutput"/>. Weight-independent: it operates
        /// only on the per-head Q/K/V buffers, so the F32 and Q8 projection paths
        /// (<see cref="Decode"/> / <see cref="DecodeQuantized"/>) share it verbatim.
        /// </summary>
        private void AttendCore(
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position,
            RopeTable? rope)
        {
            // Apply RoPE to Q and K at the current position before attention and cache write.
            // Q is rotated for the current step. K is rotated and stored — cached K vectors
            // are permanently rotated, so no re-rotation is needed at read time.
            if (rope is not null)
            {
                RopeKernel.Apply(_query, rope, position);
                RopeKernel.Apply(_key, rope, position);
            }

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
