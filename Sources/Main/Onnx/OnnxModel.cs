// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// In-memory representation of an ONNX model.
    /// Mirrors a subset of the ModelProto / GraphProto schema sufficient for inference import.
    /// </summary>
    public sealed class OnnxModel
    {
        public long IrVersion { get; init; }
        public string ProducerName { get; init; } = "";
        public string ProducerVersion { get; init; } = "";
        public List<OnnxOpsetImport> OpsetImports { get; init; } = new();
        public OnnxGraph Graph { get; init; } = new();
    }

    public sealed class OnnxOpsetImport
    {
        public string Domain { get; init; } = ""; // empty = ai.onnx (default)
        public long Version { get; init; }
    }

    public sealed class OnnxGraph
    {
        public string Name { get; init; } = "";
        public List<OnnxNode> Nodes { get; init; } = new();
        public List<OnnxValueInfo> Inputs { get; init; } = new();
        public List<OnnxValueInfo> Outputs { get; init; } = new();
        public List<OnnxTensor> Initializers { get; init; } = new();
    }

    public sealed class OnnxNode
    {
        public string Name { get; init; } = "";
        public string OpType { get; init; } = "";
        public List<string> Inputs { get; init; } = new();
        public List<string> Outputs { get; init; } = new();
        public Dictionary<string, OnnxAttribute> Attributes { get; init; } = new();
    }

    /// <summary>
    /// A single attribute on an ONNX node (e.g., "kernel_shape", "strides", "epsilon").
    /// </summary>
    public sealed class OnnxAttribute
    {
        public string Name { get; init; } = "";
        public OnnxAttributeType Type { get; init; }

        // Only one of these is populated based on Type
        public long IntValue { get; init; }
        public float FloatValue { get; init; }
        public string StringValue { get; init; } = "";
        public long[] IntArray { get; init; } = Array.Empty<long>();
        public float[] FloatArray { get; init; } = Array.Empty<float>();
        public OnnxTensor? TensorValue { get; init; }
    }

    public enum OnnxAttributeType
    {
        Undefined = 0,
        Float = 1,
        Int = 2,
        String = 3,
        Tensor = 4,
        Graph = 5,
        Floats = 6,
        Ints = 7,
        Strings = 8,
        Tensors = 9,
        Graphs = 10,
    }

    /// <summary>
    /// Tensor data, used for both initializers (model weights) and constants.
    /// </summary>
    public sealed class OnnxTensor
    {
        public string Name { get; init; } = "";
        public OnnxDataType DataType { get; init; }
        public long[] Dims { get; init; } = Array.Empty<long>();

        /// <summary>Raw bytes of tensor data, little-endian. Decoded based on DataType.</summary>
        public byte[] RawData { get; init; } = Array.Empty<byte>();

        /// <summary>Float data when stored unpacked (alternative to RawData).</summary>
        public float[]? FloatData { get; init; }

        /// <summary>Int64 data when stored unpacked.</summary>
        public long[]? Int64Data { get; init; }

        /// <summary>
        /// Set when tensor data is stored externally (PyTorch ≥ 2.x default for non-trivial models).
        /// When non-null, RawData/FloatData/Int64Data are empty and data must be loaded from
        /// the file referenced by ExternalData.Location.
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

    /// <summary>
    /// Reference to externally-stored tensor data. Resolved at load time by reading
    /// <c>Length</c> bytes from <c>Location</c> starting at <c>Offset</c>.
    /// </summary>
    public sealed class OnnxExternalDataInfo
    {
        /// <summary>Filename relative to the .onnx file's directory.</summary>
        public string Location { get; init; } = "";

        /// <summary>Byte offset into the external data file.</summary>
        public long Offset { get; init; }

        /// <summary>Number of bytes to read. May be 0 if reading whole file.</summary>
        public long Length { get; init; }
    }

    public enum OnnxDataType
    {
        Undefined = 0,
        Float = 1,
        Uint8 = 2,
        Int8 = 3,
        Uint16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        Uint32 = 12,
        Uint64 = 13,
    }

    public sealed class OnnxValueInfo
    {
        public string Name { get; init; } = "";
        public OnnxDataType DataType { get; init; }

        /// <summary>Dimensions; null entries indicate dynamic dim. Negative = symbolic dim.</summary>
        public long?[] Shape { get; init; } = Array.Empty<long?>();
    }
}
