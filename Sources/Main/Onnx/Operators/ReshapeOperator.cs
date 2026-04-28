// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx.Operators
{
    /// <summary>
    /// Maps ONNX <c>Reshape</c> and <c>Flatten</c> to <see cref="FlattenLayer"/>
    /// when the operation reduces rank to 2 ([batch, C, H, W] → [batch, C*H*W]).
    /// Returns null when rank does not change (no structural op needed).
    /// </summary>
    internal static class ReshapeOperator
    {
        public static IModule? Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var inputShape = shapes.GetShape(node.Inputs[0]);

            int[]? targetShape = null;

            // Second input is the target shape as an int64[] constant initializer.
            if (node.Inputs.Count >= 2 && initializers.TryGetValue(node.Inputs[1], out var shapeTensor))
            {
                var dims = shapeTensor.Int64Data ?? DecodeInt64Tensor(shapeTensor);

                if (dims != null)
                {
                    targetShape = new int[dims.Length];
                    for (var i = 0; i < dims.Length; i++)
                    {
                        targetShape[i] = (int)dims[i];
                    }

                    // Resolve -1 dimension (dynamic size).
                    var negIdx = -1;
                    for (var i = 0; i < targetShape.Length; i++)
                    {
                        if (targetShape[i] == -1) { negIdx = i; break; }
                    }

                    if (negIdx >= 0 && inputShape != null)
                    {
                        var totalElements = 1;
                        foreach (var d in inputShape) totalElements *= d;

                        var knownProduct = 1;
                        for (var i = 0; i < targetShape.Length; i++)
                        {
                            if (i != negIdx) knownProduct *= targetShape[i];
                        }

                        targetShape[negIdx] = totalElements / knownProduct;
                    }
                }
            }

            // Fallback: assume flatten [batch, rest].
            if (targetShape == null && inputShape != null)
            {
                var rest = 1;
                for (var i = 1; i < inputShape.Length; i++) rest *= inputShape[i];
                targetShape = [inputShape[0], rest];
            }

            if (targetShape != null)
            {
                shapes.SetShape(node.Outputs[0], targetShape);
            }

            // Same rank → no structural change needed.
            if (inputShape != null && targetShape != null && inputShape.Length == targetShape.Length)
            {
                return null;
            }

            // Rank decreases (e.g., 4D → 2D) → insert explicit FlattenLayer.
            return new FlattenLayer();
        }

        private static long[]? DecodeInt64Tensor(OnnxTensor tensor)
        {
            if (tensor.Int64Data is { Length: > 0 })
            {
                return tensor.Int64Data;
            }

            if (tensor.RawData.Length == 0 || tensor.RawData.Length % 8 != 0)
            {
                return null;
            }

            var count = tensor.RawData.Length / 8;
            var result = new long[count];
            var span = tensor.RawData.AsSpan();

            for (var i = 0; i < count; i++)
            {
                result[i] = BinaryPrimitives.ReadInt64LittleEndian(span.Slice(i * 8, 8));
            }

            return result;
        }
    }
}
