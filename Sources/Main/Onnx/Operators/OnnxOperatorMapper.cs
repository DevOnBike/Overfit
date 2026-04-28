// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.Onnx.Operators
{
    /// <summary>
    /// Maps ONNX operator nodes to Overfit <see cref="IModule"/> instances.
    /// 
    /// <para>
    /// STATUS: STUB. Per-operator handlers (Conv, Gemm, Relu, MaxPool, Reshape, Flatten)
    /// are not yet implemented. This stub allows the project to build while individual
    /// operators are added incrementally. See ONNX_IMPLEMENTATION_PLAN.md for skeletons.
    /// </para>
    /// </summary>
    internal static class OnnxOperatorMapper
    {
        public static IModule? MapNode(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapeContext)
        {
            return node.OpType switch
            {
                // No-op nodes that don't translate to a layer in Overfit's linear pipeline
                "Identity" => null,

                // Operators with implementations pending - clear error message points to plan
                "Conv" or "Gemm" or "Relu" or "MaxPool" or "Reshape" or "Flatten" or
                "AveragePool" or "GlobalAveragePool" or "BatchNormalization" or
                "Sigmoid" or "Tanh" or "Softmax" =>
                    throw new NotImplementedException(
                        $"ONNX operator '{node.OpType}' is on the supported list but the handler is not yet wired up. " +
                        $"See ONNX_IMPLEMENTATION_PLAN.md, section 'Implementation guide' for the skeleton."),

                // Truly unsupported operators
                _ => throw new NotSupportedException(
                    $"Unsupported ONNX operator: '{node.OpType}'. " +
                    $"See ONNX_IMPLEMENTATION_PLAN.md for the supported operator list and roadmap.")
            };
        }
    }
}