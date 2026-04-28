// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    public sealed class OnnxOpsetImport
    {
        /// <summary>Empty string means default ai.onnx domain.</summary>
        public string Domain { get; init; } = "";
        public long Version { get; init; }
    }
}
