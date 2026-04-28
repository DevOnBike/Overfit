// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    /// <summary>
    /// Tensor data — used for both model weight initializers and inline constants.
    /// After external data resolution, RawData always contains the decoded bytes.
    /// </summary>
    public sealed class OnnxTensor
    {
        public string Name { get; init; } = "";
        public OnnxDataType DataType { get; init; }
        public long[] Dims { get; init; } = Array.Empty<long>();

        /// <summary>Raw little-endian bytes. Populated after ResolveExternalData.</summary>
        public byte[] RawData { get; init; } = Array.Empty<byte>();

        /// <summary>Float data when stored unpacked in the protobuf (alternative to RawData).</summary>
        public float[]? FloatData { get; init; }

        /// <summary>Int64 data when stored unpacked.</summary>
        public long[]? Int64Data { get; init; }

        /// <summary>
        /// Non-null when data lives in a separate .data file (PyTorch ≥ 2.x default).
        /// Resolved to RawData by OnnxImporter before operator mapping begins.
        /// </summary>
        public OnnxExternalDataInfo? ExternalData { get; init; }

        public int ElementCount
        {
            get
            {
                long count = 1;
                foreach (var d in Dims) count *= d;
                return (int)count;
            }
        }
    }
}
