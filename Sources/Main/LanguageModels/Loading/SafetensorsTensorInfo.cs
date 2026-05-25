// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>One tensor's header entry. Byte offsets are relative to the data block.</summary>
    public readonly struct SafetensorsTensorInfo
    {
        public SafetensorsTensorInfo(string name, SafetensorsDType dtype, long[] shape, long begin, long end)
        {
            Name = name;
            DType = dtype;
            Shape = shape;
            Begin = begin;
            End = end;

            var count = 1L;
            for (var i = 0; i < shape.Length; i++) { count *= shape[i]; }
            ElementCount = shape.Length == 0 ? 0 : count;
        }

        public string Name { get; }
        public SafetensorsDType DType { get; }
        public long[] Shape { get; }
        public long Begin { get; }
        public long End { get; }
        public long ElementCount { get; }
    }
}
