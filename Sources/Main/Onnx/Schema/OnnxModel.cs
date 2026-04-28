// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
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
}
