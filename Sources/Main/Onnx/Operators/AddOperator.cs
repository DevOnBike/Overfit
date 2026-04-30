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
    /// Maps ONNX <c>Add</c> to <see cref="OnnxAddLayer"/>.
    /// Used for residual / skip connections in ResNet-style models.
    /// Both inputs must have identical shapes.
    /// </summary>
    internal static class AddOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var shape0 = shapes.GetShape(node.Inputs[0]);
            var shape1 = shapes.GetShape(node.Inputs[1]);

            if (shape0 != null)
            {
                shapes.SetShape(node.Outputs[0], shape0);
            }
            else if (shape1 != null)
            {
                shapes.SetShape(node.Outputs[0], shape1);
            }

            return new OnnxAddLayer();
        }
    }
}
