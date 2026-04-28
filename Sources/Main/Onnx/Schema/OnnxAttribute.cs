// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    /// <summary>
    /// A single attribute on an ONNX node (e.g., "kernel_shape", "strides", "epsilon").
    /// Only one value field is populated, matching the Type discriminator.
    /// </summary>
    public sealed class OnnxAttribute
    {
        public string Name { get; init; } = "";
        public OnnxAttributeType Type { get; init; }
        public long IntValue { get; init; }
        public float FloatValue { get; init; }
        public string StringValue { get; init; } = "";
        public long[] IntArray { get; init; } = Array.Empty<long>();
        public float[] FloatArray { get; init; } = Array.Empty<float>();
        public OnnxTensor? TensorValue { get; init; }
    }
}
