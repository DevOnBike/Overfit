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