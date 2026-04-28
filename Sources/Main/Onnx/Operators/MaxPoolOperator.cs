// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx.Operators
{
    /// <summary>
    /// Maps ONNX <c>MaxPool</c> to <see cref="MaxPool2DLayer"/>.
    ///
    /// MVP constraints (clear error thrown otherwise):
    ///   - Square kernel (kH == kW)
    ///   - Stride equals pool size (non-overlapping)
    ///   - Zero padding
    ///   - ceil_mode = 0
    /// </summary>
    internal static class MaxPoolOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var kernelShape = node.Attributes.TryGetValue("kernel_shape", out var ks)
                ? ks.IntArray
                : throw new InvalidDataException("MaxPool missing required attribute 'kernel_shape'.");

            var strides = node.Attributes.TryGetValue("strides", out var st)
                ? st.IntArray
                : new long[] { 1, 1 };

            var pads = node.Attributes.TryGetValue("pads", out var p)
                ? p.IntArray
                : new long[] { 0, 0, 0, 0 };

            var ceilMode = node.Attributes.TryGetValue("ceil_mode", out var cm)
                ? cm.IntValue
                : 0L;

            if (kernelShape.Length != 2 || kernelShape[0] != kernelShape[1])
            {
                throw new NotSupportedException(
                    $"MaxPool: only square kernels supported, got [{string.Join(",", kernelShape)}].");
            }

            if (strides[0] != strides[1] || strides[0] != kernelShape[0])
            {
                throw new NotSupportedException(
                    $"MaxPool: stride must equal kernel size for non-overlapping pool " +
                    $"(kernel={kernelShape[0]}, stride={strides[0]}).");
            }

            if (pads.Any(pd => pd != 0))
            {
                throw new NotSupportedException("MaxPool padding not yet supported.");
            }

            if (ceilMode != 0)
            {
                throw new NotSupportedException("MaxPool ceil_mode=1 not supported.");
            }

            var poolSize = (int)kernelShape[0];

            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new InvalidDataException(
                    $"MaxPool: input '{node.Inputs[0]}' has no known shape in context.");

            if (inputShape.Length != 4)
            {
                throw new InvalidDataException(
                    $"MaxPool: input rank should be 4, got {inputShape.Length}.");
            }

            int batch = inputShape[0], channels = inputShape[1], h = inputShape[2], w = inputShape[3];

            shapes.SetShape(node.Outputs[0], [batch, channels, h / poolSize, w / poolSize]);

            return new MaxPool2DLayer(channels, h, w, poolSize);
        }
    }
}
