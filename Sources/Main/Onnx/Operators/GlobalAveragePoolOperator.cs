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
    /// Maps ONNX <c>GlobalAveragePool</c> to <see cref="GlobalAveragePool2DLayer"/>.
    ///
    /// ONNX GlobalAveragePool:
    ///   Input:  [batch, C, H, W]  (NCHW layout, 4-D)
    ///   Output: [batch, C, 1, 1]  (ONNX spec) — flattened to [batch, C] in Overfit
    ///
    /// The output shape difference (ONNX keeps spatial dims as 1, Overfit drops them)
    /// is handled by recording [batch, C] in the shape context so downstream Gemm/Linear
    /// nodes see the correct K dimension.
    /// </summary>
    internal static class GlobalAveragePoolOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new InvalidDataException(
                    $"GlobalAveragePool: input '{node.Inputs[0]}' has no known shape.");

            if (inputShape.Length != 4)
            {
                throw new InvalidDataException(
                    $"GlobalAveragePool: expected 4-D input [N,C,H,W], got rank {inputShape.Length}.");
            }

            int batch    = inputShape[0];
            int channels = inputShape[1];
            int h        = inputShape[2];
            int w        = inputShape[3];

            // Overfit's GlobalAveragePool2DLayer outputs [batch, channels] (drops H×W).
            // ONNX spec says output is [batch, channels, 1, 1] but downstream Gemm/Linear
            // nodes expect K = channels, so we register the flattened shape.
            shapes.SetShape(node.Outputs[0], [batch, channels]);

            return new GlobalAveragePool2DLayer(channels, h, w);
        }
    }
}
