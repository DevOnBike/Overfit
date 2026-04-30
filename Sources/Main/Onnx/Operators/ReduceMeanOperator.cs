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
    /// Maps ONNX <c>ReduceMean</c> to <see cref="GlobalAveragePool2DLayer"/>.
    ///
    /// PyTorch <c>nn.AdaptiveAvgPool2d(1)</c> and <c>nn.AdaptiveAvgPool2d(output_size=1)</c>
    /// export as <c>ReduceMean</c> with axes=[2,3] (spatial dimensions), NOT as
    /// <c>GlobalAveragePool</c>. This operator handles that specific pattern.
    ///
    /// Only axes=[2,3] (or normalised [-2,-1]) on a 4-D [N,C,H,W] input is supported.
    /// Other axes combinations throw <see cref="NotSupportedException"/> with a clear message.
    /// </summary>
    internal static class ReduceMeanOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            // ── Parse axes ────────────────────────────────────────────────────
            // ONNX opset 18+: axes may come from a second input tensor, not attribute.
            // Opset < 18: axes is an attribute.
            long[]? axes = null;

            if (node.Attributes.TryGetValue("axes", out var axesAttr))
            {
                axes = axesAttr.IntArray;
            }
            else if (node.Inputs.Count >= 2 &&
                     !string.IsNullOrEmpty(node.Inputs[1]) &&
                     initializers.TryGetValue(node.Inputs[1], out var axesTensor))
            {
                // Axes are stored as Int64 tensor — read directly without float conversion.
                if (axesTensor.Int64Data != null && axesTensor.Int64Data.Length > 0)
                {
                    axes = axesTensor.Int64Data;
                }
                else if (axesTensor.RawData.Length > 0)
                {
                    // Raw little-endian int64 bytes.
                    var count = axesTensor.RawData.Length / sizeof(long);
                    axes = new long[count];
                    for (var i = 0; i < count; i++)
                    {
                        axes[i] = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(
                            axesTensor.RawData.AsSpan(i * sizeof(long), sizeof(long)));
                    }
                }
            }

            var keepDims = node.Attributes.TryGetValue("keepdims", out var kd)
                ? kd.IntValue
                : 1L;

            // ── Input shape ───────────────────────────────────────────────────
            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new InvalidDataException(
                    $"ReduceMean: input '{node.Inputs[0]}' has no known shape.");

            if (inputShape.Length != 4)
            {
                throw new NotSupportedException(
                    $"ReduceMean: only 4-D [N,C,H,W] inputs supported, got rank {inputShape.Length}. " +
                    "General ReduceMean is not implemented — only the GlobalAveragePool pattern (axes=[2,3]).");
            }

            // ── Validate axes = spatial dims [2,3] ───────────────────────────
            var rank = inputShape.Length;
            if (axes == null || axes.Length != 2)
            {
                throw new NotSupportedException(
                    $"ReduceMean: expected axes=[2,3] (GlobalAveragePool pattern), got " +
                    (axes == null ? "no axes" : $"axes=[{string.Join(",", axes)}]") + ". " +
                    "Only spatial reduction over H and W dimensions is supported.");
            }

            // Normalise negative axes.
            var norm0 = axes[0] < 0 ? axes[0] + rank : axes[0];
            var norm1 = axes[1] < 0 ? axes[1] + rank : axes[1];

            if (!((norm0 == 2 && norm1 == 3) || (norm0 == 3 && norm1 == 2)))
            {
                throw new NotSupportedException(
                    $"ReduceMean: expected axes=[2,3] (H,W), got axes=[{axes[0]},{axes[1]}] " +
                    $"(normalised: [{norm0},{norm1}]). Only GlobalAveragePool pattern is supported.");
            }

            int batch = inputShape[0];
            int channels = inputShape[1];
            int h = inputShape[2];
            int w = inputShape[3];

            // ── Output shape ──────────────────────────────────────────────────
            // keepdims=1 → [N, C, 1, 1]  (ONNX spec)
            // keepdims=0 → [N, C]         (squeezed)
            // GlobalAveragePool2DLayer outputs [N, C] (drops H,W).
            // We register the shape that downstream nodes will see.
            var outputShape = keepDims == 1
                ? new[] { batch, channels, 1, 1 }
                : new[] { batch, channels };

            shapes.SetShape(node.Outputs[0], outputShape);

            return new GlobalAveragePool2DLayer(channels, h, w);
        }
    }
}