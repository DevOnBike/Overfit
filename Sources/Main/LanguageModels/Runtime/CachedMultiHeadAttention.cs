// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Rope;

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

        /// <param name="kvHeadCount">
        /// Number of KV heads for GQA. 0 or equal to headCount = standard MHA.
        /// </param>
        public CachedMultiHeadAttention(
            int dModel,
            int headCount,
            int maxSequenceLength,
            int kvHeadCount = 0)
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

            var resolvedKvHeads = kvHeadCount > 0 ? kvHeadCount : headCount;
            if (headCount % resolvedKvHeads != 0)
            {
                throw new ArgumentException(
                $"headCount ({headCount}) must be divisible by kvHeadCount ({resolvedKvHeads}).");
            }

            DModel            = dModel;
            HeadCount         = headCount;
            KvHeadCount       = resolvedKvHeads;
            HeadDimension     = dModel / headCount;
            MaxSequenceLength = maxSequenceLength;

            _heads      = new CachedSingleHeadAttention[headCount];
            _headOutput = new float[dModel];

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
            var bo = weights.AttentionBias;
            if (bo.IsEmpty)
            {
                output.Slice(0, DModel).Clear();
            }
            else
            {
                bo.Slice(0, DModel).CopyTo(output);
            }

            var useGqa = weights.HasGqa;

            for (var h = 0; h < HeadCount; h++)
            {
                ref readonly var hw = ref weights.Head(h);

                ReadOnlySpan<float> wk, wv, bk, bv;
                if (useGqa)
                {
                    // GQA: multiple Q heads share one KV head
                    // GQA grouped mapping: Q head h uses KV head h // (nHeads/nKvHeads)
                    // (matches HuggingFace transformers repeat_kv convention)
                    var groupSize = HeadCount / KvHeadCount;
                    var kvH = h / groupSize;
                    ref readonly var kv = ref weights.KvHead(kvH);
                    wk = kv.Wk; wv = kv.Wv;
                    bk = kv.Bk; bv = kv.Bv;
                }
                else
                {
                    // Standard MHA: each Q head has its own K/V weights
                    wk = hw.Wk; wv = hw.Wv;
                    bk = hw.Bk; bv = hw.Bv;
                }

                // Map Q head index to KV cache slot (GQA: multiple Q map to same slot)
                var kvCacheHead = useGqa ? h / (HeadCount / KvHeadCount) : h;

                _heads[h].Decode(
                    hidden,
                    hw.Wq, wk, wv,
                    hw.Bq, bk, bv,
                    hw.Wo,
                    cache,
                    layerIndex,
                    kvCacheHead,
                    position,
                    _headOutput,
                    rope);

                AddInPlace(_headOutput, output, DModel);
            }
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
