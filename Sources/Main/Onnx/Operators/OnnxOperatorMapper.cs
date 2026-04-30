// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx.Operators
{
    /// <summary>
    /// Dispatches ONNX operator nodes to their corresponding Overfit layer builders.
    /// Returns null for structural no-ops that produce no IModule.
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
                "Gemm"              => GemmOperator.Build(node, initializers, shapeContext),
                "Relu"              => ReluOperator.Build(node, initializers, shapeContext),
                "MaxPool"           => MaxPoolOperator.Build(node, initializers, shapeContext),
                "Conv"              => ConvOperator.Build(node, initializers, shapeContext),
                "Reshape"           => ReshapeOperator.Build(node, initializers, shapeContext),
                "Flatten"           => ReshapeOperator.Build(node, initializers, shapeContext),
                "Tanh"              => TanhOperator.Build(node, initializers, shapeContext),
                "Sigmoid"           => SigmoidOperator.Build(node, initializers, shapeContext),
                "Softmax"           => SoftmaxOperator.Build(node, initializers, shapeContext),
                "GlobalAveragePool" => GlobalAveragePoolOperator.Build(node, initializers, shapeContext),

                // True no-ops
                "Identity"          => null,
                "Dropout"           => null, // eval-mode dropout is identity

                // Operators known to be planned but not yet wired
                "AveragePool" or "BatchNormalization" =>
                    throw new NotImplementedException(
                        $"ONNX operator '{node.OpType}' is planned but not yet implemented. " +
                        "See ONNX_IMPLEMENTATION_PLAN.md."),

                _ => throw new NotSupportedException(
                    $"Unsupported ONNX operator: '{node.OpType}'. " +
                    "Supported: Conv, Gemm, Relu, MaxPool, Reshape, Flatten, " +
                    "Tanh, Sigmoid, Softmax, GlobalAveragePool.")
            };
        }
    }
}
