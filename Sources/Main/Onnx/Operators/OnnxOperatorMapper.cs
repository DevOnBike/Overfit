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
                // Core linear algebra
                "Gemm"               => GemmOperator.Build(node, initializers, shapeContext),
                "Conv"               => ConvOperator.Build(node, initializers, shapeContext),
                "Add"                => AddOperator.Build(node, initializers, shapeContext),

                // Activations
                "Relu"               => ReluOperator.Build(node, initializers, shapeContext),
                "Tanh"               => TanhOperator.Build(node, initializers, shapeContext),
                "Sigmoid"            => SigmoidOperator.Build(node, initializers, shapeContext),
                "Softmax"            => SoftmaxOperator.Build(node, initializers, shapeContext),

                // Pooling
                "MaxPool"            => MaxPoolOperator.Build(node, initializers, shapeContext),
                "GlobalAveragePool"  => GlobalAveragePoolOperator.Build(node, initializers, shapeContext),

                // Shape ops
                "Reshape"            => ReshapeOperator.Build(node, initializers, shapeContext),
                "Flatten"            => ReshapeOperator.Build(node, initializers, shapeContext),

                // Normalisation
                "BatchNormalization" => BatchNormOperator.Build(node, initializers, shapeContext),

                // ReduceMean over spatial dims [2,3] = GlobalAveragePool
                "ReduceMean"         => ReduceMeanOperator.Build(node, initializers, shapeContext),

                // True no-ops (eval-mode semantics)
                "Identity"           => null,
                "Dropout"            => null,

                "AveragePool"        => AveragePoolOperator.Build(node, initializers, shapeContext),

                _ => throw new NotSupportedException(
                    $"Unsupported ONNX operator: '{node.OpType}'. " +
                    "Supported: Conv (with padding/stride), Gemm, Relu, Tanh, Sigmoid, Softmax, " +
                    "MaxPool, GlobalAveragePool, Reshape, Flatten, BatchNormalization.")
            };
        }
    }
}
