// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    public sealed class OnnxGraph
    {
        public string Name { get; init; } = "";
        public List<OnnxNode> Nodes { get; init; } = [];
        public List<OnnxValueInfo> Inputs { get; init; } = [];
        public List<OnnxValueInfo> Outputs { get; init; } = [];
        public List<OnnxTensor> Initializers { get; init; } = [];
    }
}
