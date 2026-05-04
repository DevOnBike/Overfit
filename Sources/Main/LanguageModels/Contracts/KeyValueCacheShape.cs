// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public readonly struct KeyValueCacheShape
    {
        public KeyValueCacheShape(
            int layerCount,
            int headCount,
            int maxSequenceLength,
            int headDimension)
        {
            LayerCount = layerCount;
            HeadCount = headCount;
            MaxSequenceLength = maxSequenceLength;
            HeadDimension = headDimension;
        }

        public int LayerCount { get; }

        public int HeadCount { get; }

        public int MaxSequenceLength { get; }

        public int HeadDimension { get; }

        public long ElementsPerCache =>
            (long)LayerCount * HeadCount * MaxSequenceLength * HeadDimension;

        public long TotalElements =>
            ElementsPerCache * 2;
    }
}