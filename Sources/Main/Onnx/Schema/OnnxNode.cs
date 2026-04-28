// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Onnx.Schema
{
    public sealed class OnnxNode
    {
        public string Name { get; init; } = "";
        public string OpType { get; init; } = "";
        public List<string> Inputs { get; init; } = new();
        public List<string> Outputs { get; init; } = new();
        public Dictionary<string, OnnxAttribute> Attributes { get; init; } = new();
    }
}
