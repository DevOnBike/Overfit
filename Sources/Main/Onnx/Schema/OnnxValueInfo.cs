// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    public sealed class OnnxValueInfo
    {
        public string Name { get; init; } = "";
        public OnnxDataType DataType { get; init; }

        /// <summary>
        /// Per-dimension sizes. Null entry = dynamic/symbolic dimension.
        /// </summary>
        public long?[] Shape { get; init; } = Array.Empty<long?>();
    }
}
