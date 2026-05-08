// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    /// <summary>
    /// Describes the dimensions of a KV-cache.
    ///
    /// For standard MHA: KvHeadCount == QHeadCount.
    /// For GQA (Llama, Mistral): KvHeadCount &lt; QHeadCount.
    ///   KV cache size is proportionally smaller.
    /// </summary>
    public readonly struct KeyValueCacheShape
    {
        public KeyValueCacheShape(
            int layerCount,
            int kvHeadCount,
            int maxSequenceLength,
            int headDimension)
        {
            LayerCount        = layerCount;
            HeadCount         = kvHeadCount;   // backward-compat alias
            KvHeadCount       = kvHeadCount;
            MaxSequenceLength = maxSequenceLength;
            HeadDimension     = headDimension;
        }

        public int LayerCount        { get; }

        /// <summary>Alias for KvHeadCount. Kept for backward compatibility.</summary>
        public int HeadCount         { get; }

        /// <summary>
        /// Number of KV heads stored per layer.
        /// For MHA equals QHeadCount. For GQA is smaller.
        /// </summary>
        public int KvHeadCount       { get; }

        public int MaxSequenceLength { get; }

        public int HeadDimension     { get; }

        public long ElementsPerCache =>
            (long)LayerCount * KvHeadCount * MaxSequenceLength * HeadDimension;

        public long TotalElements =>
            ElementsPerCache * 2;
    }
}