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
        private readonly sbyte[] _q8kInputQuants;  // Q4_K activation scratch (Q8_K-quantized)
        private readonly float[] _q8kInputScales;
        private readonly short[] _q8kInputBsums;
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

            // Q8 / Q4_K activation scratch — sized for the largest projection
            // input (dModel for Q/K/V; the O projection's input is headDim, and
            // per-head Wo can't be K-quant anyway — headDim < the 256-element
            // super-block — so the dModel size covers every K-quant projection).
            _q8InputQuants = new sbyte[dModel];
            _q8InputScales = new float[(dModel + Q8DotKernel.BlockSize - 1) / Q8DotKernel.BlockSize];
            _q8kInputQuants = new sbyte[dModel];
            _q8kInputScales = new float[(dModel + Q4KDotKernel.SuperBlockElements - 1) / Q4KDotKernel.SuperBlockElements];
            _q8kInputBsums = new short[(dModel + Q4KDotKernel.GroupSize - 1) / Q4KDotKernel.GroupSize];

            _scale = 1f / MathF.Sqrt(headDimension);
        }

        public int DModel { get; }

        public int HeadDimension { get; }

        public int MaxSequenceLength { get; }

        /// <param name="hidden">Input hidden state for the current token (length DModel).</param>
        /// <param name="wq">Query projection weights for this head.</param>
        /// <param name="wk">Key projection weights for this head.</param>
        /// <param name="wv">Value projection weights for this head.</param>
        /// <param name="bq">Query projection bias (may be empty).</param>
        /// <param name="bk">Key projection bias (may be empty).</param>
        /// <param name="bv">Value projection bias (may be empty).</param>
        /// <param name="wo">Output projection weights mapping head dimension back to DModel.</param>
        /// <param name="cache">Shared key/value cache holding past keys and values.</param>
        /// <param name="layerIndex">Index of the transformer layer this head belongs to.</param>
        /// <param name="headIndex">Index of this head within the layer.</param>
        /// <param name="position">Sequence position of the current token (0-based).</param>
        /// <param name="output">Destination span receiving this head's contribution to the output (length DModel).</param>
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
        /// Single-head decode with **per-projection dispatch** — each of Q/K/V/O
        /// picks its kernel from the weight's resident format (F32 / Q8_0 /
        /// Q4_K). The runtime calls this; a heterogeneous K-quant file (e.g.
        /// Q4_K_M with Q/K Q4_K and V Q6_K-then-Q8) is handled per projection.
        /// The older all-same-type <see cref="Decode"/> / <see cref="DecodeQuantized"/>
        /// entry points are kept for tests.
        ///
        /// <para>
        /// <paramref name="projectKv"/> lets a GQA group skip the redundant K/V
        /// projection: every Q head in a KV group shares one K/V weight set and one
        /// cache slot, so the K/V matmul + RoPE + cache write only need to run for
        /// the group's first head. The remaining heads pass <c>false</c> and read
        /// the K/V the first head already wrote — bit-identical, but the K and V
        /// projections (≈ groupSize× redundant for Qwen-style 16Q/2KV) run once per
        /// group instead of once per Q head. The first head per group (and every
        /// head under standard MHA) passes <c>true</c>.
        /// </para>
        /// </summary>
        internal void DecodeDispatched(
            ReadOnlySpan<float> hidden,
            in DecodeWeight wq,
            in DecodeWeight wk,
            in DecodeWeight wv,
            ReadOnlySpan<float> bq,
            ReadOnlySpan<float> bk,
            ReadOnlySpan<float> bv,
            in DecodeWeight wo,
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position,
            Span<float> output,
            RopeTable? rope = null,
            bool projectKv = true,
            ReadOnlySpan<sbyte> hiddenQuants = default,
            ReadOnlySpan<float> hiddenScales = default,
            ReadOnlySpan<short> hiddenBsums = default,
            bool hiddenQ8kValid = false)
        {
            if (projectKv)
            {
                ProjectKvDispatched(
                    hidden, in wk, in wv, bk, bv,
                    cache, layerIndex, headIndex, position, rope,
                    hiddenQuants, hiddenScales, hiddenBsums, hiddenQ8kValid);
            }

            ProjectQDispatched(
                hidden, in wq, bq, cache, position, rope,
                hiddenQuants, hiddenScales, hiddenBsums, hiddenQ8kValid);

            AttendAndProjectO(in wo, cache, layerIndex, headIndex, position, output);
        }

        /// <summary>
        /// Project only K/V from <paramref name="hidden"/>, RoPE-rotate K, and write both
        /// to cache slot <paramref name="headIndex"/>. Under GQA this runs once per KV
        /// group (the group's shared slot) ahead of the head-parallel Q/attend/O pass, so
        /// the bandwidth-heavy Q and O projections can fan out across <b>all</b> Q heads
        /// instead of being pinned to the few KV-group workers. Bit-identical to the K/V
        /// branch of <see cref="DecodeDispatched"/>.
        /// </summary>
        internal void ProjectKvDispatched(
            ReadOnlySpan<float> hidden,
            in DecodeWeight wk,
            in DecodeWeight wv,
            ReadOnlySpan<float> bk,
            ReadOnlySpan<float> bv,
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position,
            RopeTable? rope,
            ReadOnlySpan<sbyte> hiddenQuants,
            ReadOnlySpan<float> hiddenScales,
            ReadOnlySpan<short> hiddenBsums,
            bool hiddenQ8kValid)
        {
            ProjectHiddenDispatched(hidden, in wk, bk, _key, hiddenQuants, hiddenScales, hiddenBsums, hiddenQ8kValid);
            ProjectHiddenDispatched(hidden, in wv, bv, _value, hiddenQuants, hiddenScales, hiddenBsums, hiddenQ8kValid);

            // K is rotated and stored — cached K vectors stay permanently rotated,
            // so no re-rotation at read time.
            if (rope is not null)
            {
                RopeKernel.Apply(_key, rope, position + cache.BasePosition);
            }

            cache.WriteKey(layerIndex, headIndex, position, _key);
            cache.WriteValue(layerIndex, headIndex, position, _value);
        }

        /// <summary>
        /// Project only Q from <paramref name="hidden"/> into this head's <c>_query</c>
        /// scratch and RoPE-rotate it. Held for the subsequent
        /// <see cref="AttendAndProjectO"/> on the same instance. Bit-identical to the Q
        /// projection in <see cref="DecodeDispatched"/>.
        /// </summary>
        internal void ProjectQDispatched(
            ReadOnlySpan<float> hidden,
            in DecodeWeight wq,
            ReadOnlySpan<float> bq,
            KeyValueCache cache,
            int position,
            RopeTable? rope,
            ReadOnlySpan<sbyte> hiddenQuants,
            ReadOnlySpan<float> hiddenScales,
            ReadOnlySpan<short> hiddenBsums,
            bool hiddenQ8kValid)
        {
            ProjectHiddenDispatched(hidden, in wq, bq, _query, hiddenQuants, hiddenScales, hiddenBsums, hiddenQ8kValid);
            if (rope is not null)
            {
                RopeKernel.Apply(_query, rope, position + cache.BasePosition);
            }
        }

        /// <summary>
        /// Phase 2 of a dispatched decode: attend over the cache using this head's
        /// already-projected <c>_query</c> (from <see cref="ProjectQDispatched"/>) and
        /// project the attention output through <paramref name="wo"/> into
        /// <paramref name="output"/>.
        /// </summary>
        internal void AttendAndProjectO(
            in DecodeWeight wo,
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position,
            Span<float> output)
        {
            AttendFromCache(cache, layerIndex, headIndex, position);

            // Wo's input is the per-head attention output, not `hidden`, so it
            // always quantizes its own activation.
            ProjectSequentialDispatched(
                _attentionOutput, in wo, ReadOnlySpan<float>.Empty, output, HeadDimension, DModel);
        }

        /// <summary>
        /// Q/K/V projection from <paramref name="hidden"/> with optional reuse of a
        /// pre-quantized shared Q8_K activation. When <paramref name="sharedValid"/>
        /// is set and the weight is K-quant (Q4_K / Q6_K), the shared Q8_K scratch
        /// is used directly — no per-call quantize. Q8_0 (different activation
        /// format) and F32 always read/quantize their own input.
        /// </summary>
        private void ProjectHiddenDispatched(
            ReadOnlySpan<float> hidden,
            in DecodeWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            ReadOnlySpan<sbyte> sharedQuants,
            ReadOnlySpan<float> sharedScales,
            ReadOnlySpan<short> sharedBsums,
            bool sharedValid)
        {
            if (weight.IsQ6K)
            {
                if (sharedValid)
                {
                    Q6KDotKernel.ProjectPreQuantized(
                        weight.Quantized6K, bias, output, sharedQuants, sharedScales, sharedBsums);
                }
                else
                {
                    Q6KDotKernel.Project(
                        hidden, weight.Quantized6K, bias, output,
                        _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
                }
            }
            else if (weight.IsQ4K)
            {
                if (sharedValid)
                {
                    Q4KDotKernel.ProjectPreQuantized(
                        weight.Quantized4K, bias, output, sharedQuants, sharedScales, sharedBsums);
                }
                else
                {
                    Q4KDotKernel.Project(
                        hidden, weight.Quantized4K, bias, output,
                        _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
                }
            }
            else if (weight.IsQuantized)
            {
                Q8DotKernel.Project(
                    hidden, weight.Quantized, bias, output, _q8InputQuants, _q8InputScales);
            }
            else
            {
                SingleTokenProjectionKernel.Project(
                    hidden, weight.F32, bias, output, DModel, HeadDimension);
            }
        }

        /// <summary>
        /// Per-weight projection dispatch (sequential — we are already inside
        /// the head-parallel <c>OverfitParallelFor</c>, so nested parallelism
        /// would be wrong). Picks the kernel matching the weight's resident
        /// format; Q-paths use the head's owned activation scratch.
        /// </summary>
        private void ProjectSequentialDispatched(
            ReadOnlySpan<float> input,
            in DecodeWeight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize)
        {
            if (weight.IsQ6K)
            {
                Q6KDotKernel.Project(
                    input, weight.Quantized6K, bias, output,
                    _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
            }
            else if (weight.IsQ4K)
            {
                Q4KDotKernel.Project(
                    input, weight.Quantized4K, bias, output,
                    _q8kInputQuants, _q8kInputScales, _q8kInputBsums);
            }
            else if (weight.IsQuantized)
            {
                Q8DotKernel.Project(
                    input, weight.Quantized, bias, output,
                    _q8InputQuants, _q8InputScales);
            }
            else
            {
                SingleTokenProjectionKernel.Project(
                    input, weight.F32, bias, output, inputSize, outputSize);
            }
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
                RopeKernel.Apply(_query, rope, position + cache.BasePosition);
                RopeKernel.Apply(_key, rope, position + cache.BasePosition);
            }

            // Copy in F32 mode, quantize in Q8 mode — uniform across both KV-cache dtypes.
            cache.WriteKey(layerIndex, headIndex, position, _key);
            cache.WriteValue(layerIndex, headIndex, position, _value);

            AttendFromCache(cache, layerIndex, headIndex, position);
        }

        /// <summary>
        /// Reads the cached K/V range <c>[0..position]</c> and computes this head's
        /// attention into <see cref="_attentionOutput"/>. Assumes the current
        /// position's K/V are already written to the cache slot (by this head, or
        /// — under GQA — by the group's first head) and that <see cref="_query"/>
        /// is already RoPE-rotated.
        /// </summary>
        private void AttendFromCache(
            KeyValueCache cache,
            int layerIndex,
            int headIndex,
            int position)
        {
            var sequenceLength = position + 1;

            // Q8 KV cache: read int8 K/V + per-vector scales directly (no F32 expansion → keeps decode
            // 0-alloc and cuts attention read traffic ~4×). F32 cache: the bit-identical full-precision path.
            if (cache.IsQuantized)
            {
                CachedAttentionKernel.ComputeSingleHeadQ8(
                    _query,
                    cache.GetKeyQuants(layerIndex, headIndex, fromPosition: 0, length: sequenceLength),
                    cache.GetKeyScales(layerIndex, headIndex, fromPosition: 0, length: sequenceLength),
                    cache.GetValueQuants(layerIndex, headIndex, fromPosition: 0, length: sequenceLength),
                    cache.GetValueScales(layerIndex, headIndex, fromPosition: 0, length: sequenceLength),
                    _attentionOutput,
                    _scoreScratch,
                    sequenceLength,
                    HeadDimension,
                    _scale);
                return;
            }

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
