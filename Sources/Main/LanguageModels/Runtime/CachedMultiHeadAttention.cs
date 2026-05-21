// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.Runtime;

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
    public sealed unsafe class CachedMultiHeadAttention
    {
        private readonly CachedSingleHeadAttention[] _heads;
        private readonly float[] _headOutputs;

        // Shared Q8_K quantization of the layer's `hidden` row. Q/K/V across every
        // head project the same `hidden`, so under a K-quant attention path it is
        // quantized once here (before the head-parallel loop) and only read by all
        // heads — removing the per-head, per-projection re-quantize.
        private readonly sbyte[] _hiddenQuants;
        private readonly float[] _hiddenScales;
        private readonly short[] _hiddenBsums;

        /// <param name="kvHeadCount">
        /// Number of KV heads for GQA. 0 or equal to headCount = standard MHA.
        /// </param>
        public CachedMultiHeadAttention(
            int dModel,
            int headCount,
            int maxSequenceLength,
            int kvHeadCount = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
            if (dModel % headCount != 0)
            {
                throw new ArgumentException(
                $"dModel ({dModel}) must be divisible by headCount ({headCount}).",
                nameof(dModel));
            }
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);

            var resolvedKvHeads = kvHeadCount > 0 ? kvHeadCount : headCount;
            if (headCount % resolvedKvHeads != 0)
            {
                throw new ArgumentException(
                $"headCount ({headCount}) must be divisible by kvHeadCount ({resolvedKvHeads}).");
            }

            DModel = dModel;
            HeadCount = headCount;
            KvHeadCount = resolvedKvHeads;
            HeadDimension = dModel / headCount;
            MaxSequenceLength = maxSequenceLength;

            _heads = new CachedSingleHeadAttention[headCount];
            _headOutputs = new float[headCount * dModel];

            _hiddenQuants = new sbyte[dModel];
            _hiddenScales = new float[(dModel + Q4KDotKernel.SuperBlockElements - 1) / Q4KDotKernel.SuperBlockElements];
            _hiddenBsums = new short[(dModel + Q4KDotKernel.GroupSize - 1) / Q4KDotKernel.GroupSize];

            for (var h = 0; h < headCount; h++)
            {
                _heads[h] = new CachedSingleHeadAttention(
                dModel, HeadDimension, maxSequenceLength);
            }
        }

        public int DModel { get; }

        public int HeadCount { get; }

        public int HeadDimension { get; }

        public int MaxSequenceLength { get; }

        /// <summary>Number of KV heads. Equal to HeadCount for MHA, less for GQA.</summary>
        public int KvHeadCount { get; }

        internal void Decode(
            ReadOnlySpan<float> hidden,
            in BlockWeights weights,
            KeyValueCache cache,
            int layerIndex,
            int position,
            Span<float> output,
            RopeTable? rope = null)
        {
            // Output starts as the attention bias (or zero); each head's
            // projected contribution is summed in afterwards.
            var bo = weights.AttentionBias;
            if (bo.IsEmpty)
            {
                output.Slice(0, DModel).Clear();
            }
            else
            {
                bo.Slice(0, DModel).CopyTo(output);
            }

            // Q/K/V across every head project the same `hidden` row. When the
            // attention weights are K-quant (Q4_K / Q6_K both consume the Q8_K
            // activation form), quantize `hidden` once here and let every head
            // reuse it — instead of re-quantizing per head, per projection. Peek
            // head 0's Wq: a layer's Q/K/V share a residency.
            ref readonly var head0 = ref weights.Head(0);
            var hiddenQ8kValid = head0.Wq.IsQ4K || head0.Wq.IsQ6K;
            if (hiddenQ8kValid)
            {
                Q4KDotKernel.QuantizeActivationQ8K(
                    hidden.Slice(0, DModel), _hiddenQuants, _hiddenScales, _hiddenBsums);
            }

            // Heads are parallelised across KV groups via OverfitParallelFor:
            // one worker per KV head, each owning a disjoint KV-cache slot and a
            // disjoint contiguous run of Q heads. No cross-worker writes — the
            // result is bit-identical to the sequential head loop. Each head
            // writes its own band of _headOutputs; the reduction runs after.
            fixed (float* hiddenPtr = hidden)
            fixed (sbyte* hiddenQuantsPtr = _hiddenQuants)
            fixed (float* hiddenScalesPtr = _hiddenScales)
            fixed (short* hiddenBsumsPtr = _hiddenBsums)
            {
                var context = new HeadDecodeContext
                {
                    Heads = _heads,
                    Weights = weights,
                    Cache = cache,
                    Rope = rope,
                    HeadOutputs = _headOutputs,
                    Hidden = hiddenPtr,
                    HiddenQuants = hiddenQuantsPtr,
                    HiddenScales = hiddenScalesPtr,
                    HiddenBsums = hiddenBsumsPtr,
                    HiddenScalesLength = _hiddenScales.Length,
                    HiddenBsumsLength = _hiddenBsums.Length,
                    HiddenQ8kValid = hiddenQ8kValid,
                    LayerIndex = layerIndex,
                    Position = position,
                    DModel = DModel,
                    HeadCount = HeadCount,
                    KvHeadCount = KvHeadCount,
                    UseGqa = weights.HasGqa,
                };

                var contextPtr = Unsafe.AsPointer(ref context);

                if (KvHeadCount > 1 && OverfitParallelFor.WorkerCount > 1)
                {
                    OverfitParallelFor.For(0, KvHeadCount, &DecodeKvGroup, contextPtr);
                }
                else
                {
                    DecodeKvGroup(0, KvHeadCount, contextPtr);
                }
            }

            // Reduce: output += Σ head outputs, ascending — matches the
            // sequential accumulation order exactly.
            for (var h = 0; h < HeadCount; h++)
            {
                AddInPlace(_headOutputs.AsSpan(h * DModel, DModel), output, DModel);
            }
        }

        /// <summary>
        /// Batched (prefill) multi-head attention — scores <paramref name="rows"/>
        /// query positions (the prompt tokens at cache positions
        /// <c>basePosition .. basePosition+rows-1</c>) through every head in one pass,
        /// the multi-row counterpart of <see cref="Decode"/>.
        ///
        /// <para>
        /// Per head: batched Q/K/V projection (<see cref="BatchedProjectionKernel"/>),
        /// per-position cache writes, batched causal attention
        /// (<see cref="BatchedAttentionKernel"/> — query <c>n</c> attends
        /// <c>[0..basePosition+n]</c>), batched O projection accumulated into
        /// <paramref name="output"/> in head order — so the result is
        /// <b>bit-identical</b> to running <see cref="Decode"/> per token. Scoped to
        /// the F32 / GPT-2 path (no RoPE, standard MHA — not GQA, F32 weights);
        /// throws for the quantized / RoPE / GQA cases (the follow-on quant path).
        /// The cache must already be advanced to length <c>basePosition+rows</c>.
        /// </para>
        /// </summary>
        internal void DecodeBatched(
            ReadOnlySpan<float> hidden,
            int rows,
            in BlockWeights weights,
            KeyValueCache cache,
            int layerIndex,
            int basePosition,
            Span<float> output,
            RopeTable? rope = null)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);

            if (rope is not null)
            {
                throw new NotSupportedException("Batched prefill does not support RoPE yet (F32/GPT-2 path only).");
            }
            if (weights.HasGqa)
            {
                throw new NotSupportedException("Batched prefill does not support GQA yet (standard MHA only).");
            }

            var dModel = DModel;
            var headDim = HeadDimension;
            var cacheLength = basePosition + rows;
            var scale = 1f / MathF.Sqrt(headDim);

            // Output starts as the attention bias (or zero) per row; each head's
            // projected contribution is summed in afterwards (ascending head order,
            // matching the single-token reduction).
            var bias = weights.AttentionBias;
            for (var n = 0; n < rows; n++)
            {
                var outRow = output.Slice(n * dModel, dModel);
                if (bias.IsEmpty)
                {
                    outRow.Clear();
                }
                else
                {
                    bias.Slice(0, dModel).CopyTo(outRow);
                }
            }

            ref readonly var head0 = ref weights.Head(0);
            if (head0.Wq.IsQuantized || head0.Wq.IsQ4K || head0.Wq.IsQ6K)
            {
                throw new NotSupportedException("Batched prefill supports F32 attention weights only (quantized path is a follow-on).");
            }

            // Parallelise OVER HEADS (one dispatch for the whole layer), each head
            // sequential internally — mirrors the single-token head-parallel path.
            // Per-head scratch + output bands keep heads disjoint (no cross-head
            // write race), then the bands are reduced into output in ascending head
            // order. (The earlier per-head ProjectParallel fan-out — ~60 tiny
            // dispatches/layer — measured 3.8× SLOWER; this restructure removes it.)
            var q = new float[HeadCount * rows * headDim];
            var k = new float[HeadCount * rows * headDim];
            var v = new float[HeadCount * rows * headDim];
            var attn = new float[HeadCount * rows * headDim];
            var bands = new float[HeadCount * rows * dModel];
            var scoreScratch = new float[HeadCount * rows * cacheLength];

            fixed (float* hiddenPtr = hidden)
            {
                var context = new BatchedHeadContext
                {
                    Weights = weights,
                    Cache = cache,
                    Hidden = hiddenPtr,
                    Q = q,
                    K = k,
                    V = v,
                    Attn = attn,
                    Bands = bands,
                    ScoreScratch = scoreScratch,
                    Rows = rows,
                    DModel = dModel,
                    HeadDim = headDim,
                    CacheLength = cacheLength,
                    LayerIndex = layerIndex,
                    BasePos = basePosition,
                    Scale = scale,
                };

                var contextPtr = Unsafe.AsPointer(ref context);

                if (HeadCount > 1 && OverfitParallelFor.WorkerCount > 1)
                {
                    OverfitParallelFor.For(0, HeadCount, &ProcessHeadRangeBatched, contextPtr);
                }
                else
                {
                    ProcessHeadRangeBatched(0, HeadCount, contextPtr);
                }
            }

            // Reduce: output += Σ head bands, ascending head order — matches the
            // single-token reduction exactly.
            for (var h = 0; h < HeadCount; h++)
            {
                for (var n = 0; n < rows; n++)
                {
                    var outRow = output.Slice(n * dModel, dModel);
                    TensorPrimitives.Add(outRow, bands.AsSpan((h * rows + n) * dModel, dModel), outRow);
                }
            }
        }

        /// <summary>Worker for <see cref="DecodeBatched"/> — one disjoint range of heads.</summary>
        private static void ProcessHeadRangeBatched(int headStart, int headEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<BatchedHeadContext>(context);

            var rows = ctx.Rows;
            var dModel = ctx.DModel;
            var headDim = ctx.HeadDim;
            var cacheLength = ctx.CacheLength;
            var layer = ctx.LayerIndex;
            var basePos = ctx.BasePos;
            var scale = ctx.Scale;

            var hidden = new ReadOnlySpan<float>(ctx.Hidden, rows * dModel);

            for (var h = headStart; h < headEnd; h++)
            {
                ref readonly var hw = ref ctx.Weights.Head(h);

                var qh = ctx.Q.AsSpan(h * rows * headDim, rows * headDim);
                var kh = ctx.K.AsSpan(h * rows * headDim, rows * headDim);
                var vh = ctx.V.AsSpan(h * rows * headDim, rows * headDim);
                var attnh = ctx.Attn.AsSpan(h * rows * headDim, rows * headDim);
                var scoreh = ctx.ScoreScratch.AsSpan(h * rows * cacheLength, rows * cacheLength);
                var bandh = ctx.Bands.AsSpan(h * rows * dModel, rows * dModel);

                // Sequential per-head projections (parallelism is across heads above).
                BatchedProjectionKernel.Project(hidden, rows, hw.Wq.F32, hw.Bq, qh, dModel, headDim);
                BatchedProjectionKernel.Project(hidden, rows, hw.Wk.F32, hw.Bk, kh, dModel, headDim);
                BatchedProjectionKernel.Project(hidden, rows, hw.Wv.F32, hw.Bv, vh, dModel, headDim);

                for (var n = 0; n < rows; n++)
                {
                    kh.Slice(n * headDim, headDim).CopyTo(ctx.Cache.GetKeyWriteSpan(layer, h, basePos + n));
                    vh.Slice(n * headDim, headDim).CopyTo(ctx.Cache.GetValueWriteSpan(layer, h, basePos + n));
                }

                var keys = ctx.Cache.GetKeyReadSpan(layer, h, fromPosition: 0, length: cacheLength);
                var values = ctx.Cache.GetValueReadSpan(layer, h, fromPosition: 0, length: cacheLength);

                BatchedAttentionKernel.Compute(qh, keys, values, attnh, scoreh, rows, cacheLength, headDim, scale);

                BatchedProjectionKernel.Project(attnh, rows, hw.Wo.F32, [], bandh, headDim, dModel);
            }
        }

        /// <summary>State for <see cref="ProcessHeadRangeBatched"/> via a stack pointer.</summary>
        private struct BatchedHeadContext
        {
            public BlockWeights Weights;
            public KeyValueCache Cache;
            public float* Hidden;
            public float[] Q;
            public float[] K;
            public float[] V;
            public float[] Attn;
            public float[] Bands;
            public float[] ScoreScratch;
            public int Rows;
            public int DModel;
            public int HeadDim;
            public int CacheLength;
            public int LayerIndex;
            public int BasePos;
            public float Scale;
        }

        /// <summary>
        /// Worker body: decodes every Q head in KV groups
        /// <c>[groupStart, groupEnd)</c> into its <c>_headOutputs</c> band. Each
        /// group owns exactly one KV-cache head — disjoint across workers, so
        /// there is no cross-worker cache write.
        /// </summary>
        private static void DecodeKvGroup(int groupStart, int groupEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<HeadDecodeContext>(context);

            var dModel = ctx.DModel;
            var groupSize = ctx.HeadCount / ctx.KvHeadCount;
            var hidden = new ReadOnlySpan<float>(ctx.Hidden, dModel);

            // Shared pre-quantized `hidden` (Q8_K) — read-only across all heads.
            var hQuants = ctx.HiddenQ8kValid
                ? new ReadOnlySpan<sbyte>(ctx.HiddenQuants, dModel)
                : default;
            var hScales = ctx.HiddenQ8kValid
                ? new ReadOnlySpan<float>(ctx.HiddenScales, ctx.HiddenScalesLength)
                : default;
            var hBsums = ctx.HiddenQ8kValid
                ? new ReadOnlySpan<short>(ctx.HiddenBsums, ctx.HiddenBsumsLength)
                : default;

            for (var group = groupStart; group < groupEnd; group++)
            {
                for (var headInGroup = 0; headInGroup < groupSize; headInGroup++)
                {
                    var h = group * groupSize + headInGroup;
                    ref readonly var hw = ref ctx.Weights.Head(h);

                    DecodeWeight wk, wv;
                    ReadOnlySpan<float> bk, bv;
                    if (ctx.UseGqa)
                    {
                        // GQA: every Q head in the group shares one KV head.
                        ref readonly var kv = ref ctx.Weights.KvHead(group);
                        wk = kv.Wk; wv = kv.Wv;
                        bk = kv.Bk; bv = kv.Bv;
                    }
                    else
                    {
                        // Standard MHA: each Q head has its own K/V weights.
                        wk = hw.Wk; wv = hw.Wv;
                        bk = hw.Bk; bv = hw.Bv;
                    }

                    var headOutput = ctx.HeadOutputs.AsSpan(h * dModel, dModel);

                    // Under GQA every Q head in the group shares one K/V weight set
                    // and one cache slot, so only the group's first head projects +
                    // writes K/V; the rest read what it wrote (bit-identical). Under
                    // standard MHA each head owns its K/V, so all project.
                    var projectKv = !ctx.UseGqa || headInGroup == 0;

                    // Per-projection dispatch (step 3.2a) — each of Q/K/V/O picks
                    // its kernel from its resident format inside DecodeDispatched.
                    // Handles heterogeneous K-quant files where Q/K/V/O may mix
                    // F32 / Q8_0 / Q4_K / Q6_K.
                    ctx.Heads[h].DecodeDispatched(
                        hidden,
                        hw.Wq, wk, wv,
                        hw.Bq, bk, bv,
                        hw.Wo,
                        ctx.Cache,
                        ctx.LayerIndex,
                        group,            // KV-cache slot for this group
                        ctx.Position,
                        headOutput,
                        ctx.Rope,
                        projectKv,
                        hQuants,
                        hScales,
                        hBsums,
                        ctx.HiddenQ8kValid);
                }
            }
        }

        /// <summary>State handed to <see cref="DecodeKvGroup"/> workers via a stack pointer.</summary>
        private struct HeadDecodeContext
        {
            public CachedSingleHeadAttention[] Heads;
            public BlockWeights Weights;
            public KeyValueCache Cache;
            public RopeTable? Rope;
            public float[] HeadOutputs;
            public float* Hidden;
            public sbyte* HiddenQuants;
            public float* HiddenScales;
            public short* HiddenBsums;
            public int HiddenScalesLength;
            public int HiddenBsumsLength;
            public bool HiddenQ8kValid;
            public int LayerIndex;
            public int Position;
            public int DModel;
            public int HeadCount;
            public int KvHeadCount;
            public bool UseGqa;
        }



        public CachedSingleHeadAttention GetHeadDecoder(int headIndex)
        {
            if ((uint)headIndex >= (uint)HeadCount)
            {
                throw new ArgumentOutOfRangeException(nameof(headIndex));
            }

            return _heads[headIndex];
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
