// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Batched single-head causal attention (prefill Phase 2): scores <c>N</c> query
    /// positions against a shared K/V cache in one call, the multi-token counterpart
    /// of <see cref="CachedAttentionKernel.ComputeSingleHead"/>.
    ///
    /// <para>
    /// The <c>N</c> queries are the most recent <c>N</c> cache entries — query
    /// <c>i</c> sits at absolute position <c>basePos + i</c> where
    /// <c>basePos = cacheLength - rows</c>, and under the causal mask attends over
    /// keys/values <c>[0 .. basePos + i]</c> (i.e. the prefix plus itself, never a
    /// later prompt token). So each query reduces to exactly one
    /// <see cref="CachedAttentionKernel.ComputeSingleHead"/> with
    /// <c>sequenceLength = basePos + i + 1</c> — making the batched output
    /// <b>bit-identical</b> to running the single-token kernel per query.
    /// </para>
    ///
    /// <para>
    /// Attention is not weight-bound (no large weight matrix — just Q·K, softmax and
    /// softmax·V over the cache), so the batched win is parallelism: the <c>N</c>
    /// queries are independent (disjoint output rows + a disjoint score-scratch row
    /// each, read-only shared K/V), so they fan out across <c>OverfitParallel</c>
    /// with no cross-worker writes and no per-query dispatch overhead.
    /// </para>
    /// </summary>
    public static unsafe class BatchedAttentionKernel
    {
        /// <summary>
        /// Whether <see cref="ComputeParallel"/> hands workers an interleaved slot order instead of the raw
        /// query order. A/B switch for the measurement below; set <c>OVERFIT_BALANCED_ATTN=0</c> to disable.
        /// </summary>
        internal static bool UseBalancedQueryOrder =
            Environment.GetEnvironmentVariable(OverfitEnvironment.BalancedAttention) != "0";

        /// <summary>
        /// Sequential batched attention. <paramref name="query"/> is row-major
        /// <c>[rows × headDim]</c>, <paramref name="keys"/>/<paramref name="values"/>
        /// row-major <c>[cacheLength × headDim]</c>, <paramref name="output"/>
        /// <c>[rows × headDim]</c>, <paramref name="scoreScratch"/>
        /// <c>[rows × cacheLength]</c> (one disjoint row per query).
        /// </summary>
        public static void Compute(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int rows,
            int cacheLength,
            int headDimension,
            float scale)
        {
            Validate(query, keys, values, output, scoreScratch, rows, cacheLength, headDimension);

            var basePos = cacheLength - rows;

            for (var i = 0; i < rows; i++)
            {
                ComputeQuery(query, keys, values, output, scoreScratch, i, basePos, cacheLength, headDimension, scale);
            }
        }

        /// <summary>
        /// Parallel batched attention — the <c>rows</c> queries fan out across the
        /// zero-allocation <c>OverfitParallel</c> pool. Bit-identical to
        /// <see cref="Compute"/> (each query is independent: disjoint output row,
        /// disjoint score-scratch row, read-only shared K/V).
        /// </summary>
        public static void ComputeParallel(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int rows,
            int cacheLength,
            int headDimension,
            float scale,
            float softcap = 0f)
        {
            Validate(query, keys, values, output, scoreScratch, rows, cacheLength, headDimension);

            fixed (float* queryPtr = query)
            fixed (float* keysPtr = keys)
            fixed (float* valuesPtr = values)
            fixed (float* outputPtr = output)
            fixed (float* scratchPtr = scoreScratch)
            {
                var context = new AttnContext
                {
                    Query = queryPtr,
                    Keys = keysPtr,
                    Values = valuesPtr,
                    Output = outputPtr,
                    Scratch = scratchPtr,
                    Rows = rows,
                    CacheLength = cacheLength,
                    HeadDimension = headDimension,
                    Scale = scale,
                    Softcap = softcap,
                    BasePos = cacheLength - rows,
                    Balanced = UseBalancedQueryOrder,
                };

                OverfitParallel.For(0, rows, &ComputeQueryRange, &context);
            }
        }

        private static void ComputeQueryRange(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<AttnContext>(context);

            var headDim = ctx.HeadDimension;
            var cacheLength = ctx.CacheLength;

            var query = new ReadOnlySpan<float>(ctx.Query, ctx.Rows * headDim);
            var keys = new ReadOnlySpan<float>(ctx.Keys, cacheLength * headDim);
            var values = new ReadOnlySpan<float>(ctx.Values, cacheLength * headDim);
            var output = new Span<float>(ctx.Output, ctx.Rows * headDim);
            var scratch = new Span<float>(ctx.Scratch, ctx.Rows * cacheLength);

            var balanced = ctx.Balanced;
            var rows = ctx.Rows;

            for (var slot = start; slot < end; slot++)
            {
                var i = balanced ? BalancedQueryIndex(slot, rows) : slot;

                ComputeQuery(query, keys, values, output, scratch, i, ctx.BasePos, cacheLength, headDim, ctx.Scale, ctx.Softcap);
            }
        }

        /// <summary>
        /// Maps a scheduling slot to a query index so that <b>any contiguous run of slots carries the same
        /// amount of work</b>, by pairing the cheapest remaining query with the most expensive one.
        ///
        /// <para><b>Why this is needed.</b> Under the causal mask query <c>i</c> attends over <c>basePos+i+1</c>
        /// keys, so work grows linearly with the query index — yet <c>OverfitParallel.For</c> splits its range
        /// into <i>contiguous</i> chunks (<c>perChunk = ceil(total/chunks)</c>). On a 672-token prefill across
        /// 32 workers that gave worker 0 rows 0-20 (≈231 dot products) and worker 31 rows 651-671 (≈13 902).
        /// The region's duration is its longest chunk, so it ran <b>1.97× longer than the balanced ideal</b>
        /// of 7 067 while 31 workers sat idle.</para>
        ///
        /// <para>Slot <c>2k</c> maps to query <c>k</c> and slot <c>2k+1</c> to query <c>rows-1-k</c>, so each
        /// consecutive pair costs <c>rows+1</c> regardless of where it lands. The mapping is a bijection over
        /// <c>[0, rows)</c> for both parities of <c>rows</c>.</para>
        ///
        /// <para>Output is <b>bit-identical</b>: queries are independent — each writes its own disjoint output
        /// row and score-scratch row, and no value is reduced across queries — so reordering them changes
        /// nothing but which worker runs which.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int BalancedQueryIndex(int slot, int rows)
        {
            var half = slot >> 1;

            return (slot & 1) == 0 ? half : rows - 1 - half;
        }

        /// <summary>
        /// One query: causal-visible length <c>basePos + i + 1</c>, delegating to the
        /// proven single-token kernel against its own output/scratch rows.
        /// </summary>
        private static void ComputeQuery(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int i,
            int basePos,
            int cacheLength,
            int headDimension,
            float scale,
            float softcap = 0f)
        {
            var visibleLength = basePos + i + 1;

            CachedAttentionKernel.ComputeSingleHead(
                query.Slice(i * headDimension, headDimension),
                keys,
                values,
                output.Slice(i * headDimension, headDimension),
                scoreScratch.Slice(i * cacheLength, cacheLength),
                visibleLength,
                headDimension,
                scale,
                softcap);
        }

        private static void Validate(
            ReadOnlySpan<float> query,
            ReadOnlySpan<float> keys,
            ReadOnlySpan<float> values,
            Span<float> output,
            Span<float> scoreScratch,
            int rows,
            int cacheLength,
            int headDimension)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

            if (rows > cacheLength)
            {
                throw new ArgumentException($"rows ({rows}) cannot exceed cacheLength ({cacheLength}).", nameof(rows));
            }
            if (query.Length < (long)rows * headDimension)
            {
                throw new ArgumentException("Query span < rows*headDimension.", nameof(query));
            }
            if (output.Length < (long)rows * headDimension)
            {
                throw new ArgumentException("Output span < rows*headDimension.", nameof(output));
            }
            if (keys.Length < (long)cacheLength * headDimension)
            {
                throw new ArgumentException("Keys span < cacheLength*headDimension.", nameof(keys));
            }
            if (values.Length < (long)cacheLength * headDimension)
            {
                throw new ArgumentException("Values span < cacheLength*headDimension.", nameof(values));
            }
            if (scoreScratch.Length < (long)rows * cacheLength)
            {
                throw new ArgumentException("Score-scratch span < rows*cacheLength.", nameof(scoreScratch));
            }
        }

        private struct AttnContext
        {
            public float* Query;
            public float* Keys;
            public float* Values;
            public float* Output;
            public float* Scratch;
            public int Rows;
            public int CacheLength;
            public int HeadDimension;
            public float Scale;
            public float Softcap;
            public int BasePos;
            public bool Balanced;
        }
    }
}
