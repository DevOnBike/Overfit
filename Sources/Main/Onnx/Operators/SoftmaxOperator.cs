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
    /// Maps ONNX <c>Softmax</c> to <see cref="SoftmaxActivation"/>.
    ///
    /// ONNX Softmax applies along the last axis (axis=-1) by default.
    /// Only axis=-1 and axis equal to the last dimension are supported — this is
    /// the common case for classification output layers. Other axes throw.
    /// </summary>
    internal static class SoftmaxOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var axis = node.Attributes.TryGetValue("axis", out var axisAttr)
                ? axisAttr.IntValue
                : -1L;

            var inputShape = shapes.GetShape(node.Inputs[0]);

            if (inputShape != null)
            {
                // Normalise negative axis: -1 == last dimension.
                var rank = inputShape.Length;
                var normalisedAxis = axis < 0 ? axis + rank : axis;

                if (normalisedAxis != rank - 1)
                {
                    throw new NotSupportedException(
                        $"Softmax: only axis=-1 (last dimension) is supported, got axis={axis} " +
                        $"(rank={rank}, normalised={normalisedAxis}).");
                }

                shapes.SetShape(node.Outputs[0], inputShape);
            }

            return new SoftmaxActivation();
        }
    }
}
