// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    /// <summary>
    /// Reference to tensor data stored in a separate file alongside the .onnx model.
    /// PyTorch ≥ 2.x emits external data by default even for small models.
    /// </summary>
    public sealed class OnnxExternalDataInfo
    {
        /// <summary>Filename relative to the .onnx file's directory.</summary>
        public string Location { get; init; } = "";

        /// <summary>Byte offset into the external data file.</summary>
        public long Offset { get; init; }

        /// <summary>Number of bytes to read. Zero means read to end of file.</summary>
        public long Length { get; init; }
    }
}
