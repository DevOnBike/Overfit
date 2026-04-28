// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx.Operators
{
    /// <summary>
    /// Maps ONNX <c>Relu</c> to <see cref="ReluActivation"/>.
    /// No attributes, no weights — just propagates shape.
    /// </summary>
    internal static class ReluOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var inputShape = shapes.GetShape(node.Inputs[0]);
            if (inputShape != null)
            {
                shapes.SetShape(node.Outputs[0], inputShape);
            }

            return new ReluActivation();
        }
    }
}
