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
    /// Maps ONNX <c>MaxPool</c> to <see cref="MaxPool2DLayer"/>.
    ///
    /// Supported (clear error thrown otherwise):
    ///   - Square kernel (kH == kW) and square stride
    ///   - Overlapping pools (stride may be &lt; kernel, e.g. ResNet's 3x3 stride-2)
    ///   - Symmetric square zero-padding (all four pads equal, e.g. ResNet's [1,1,1,1])
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
                : throw new OverfitFormatException("MaxPool missing required attribute 'kernel_shape'.");

            var strides = node.Attributes.TryGetValue("strides", out var st)
                ? st.IntArray
                : [1, 1];

            var pads = node.Attributes.TryGetValue("pads", out var p)
                ? p.IntArray
                : [0, 0, 0, 0];

            var ceilMode = node.Attributes.TryGetValue("ceil_mode", out var cm)
                ? cm.IntValue
                : 0L;

            if (kernelShape.Length != 2 || kernelShape[0] != kernelShape[1])
            {
                throw new OverfitRuntimeException(
                    $"MaxPool: only square kernels supported, got [{string.Join(",", kernelShape)}].");
            }

            if (strides[0] != strides[1])
            {
                throw new OverfitRuntimeException(
                    $"MaxPool: only square strides supported, got [{string.Join(",", strides)}].");
            }

            // ONNX pads = [H_begin, W_begin, H_end, W_end]. We support fully-symmetric square padding (all
            // four equal — covers ResNet's [1,1,1,1]); the kernel pads with -inf, so it just widens the window.
            var padding = pads.Length > 0 ? (int)pads[0] : 0;
            for (var pi = 0; pi < pads.Length; pi++)
            {
                if (pads[pi] != padding)
                {
                    throw new OverfitRuntimeException(
                        $"MaxPool: only symmetric square padding supported (all pads equal), got [{string.Join(",", pads)}].");
                }
            }

            if (ceilMode != 0)
            {
                throw new OverfitRuntimeException("MaxPool ceil_mode=1 not supported.");
            }

            var poolSize = (int)kernelShape[0];
            var stride = (int)strides[0];

            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new OverfitFormatException(
                    $"MaxPool: input '{node.Inputs[0]}' has no known shape in context.");

            if (inputShape.Length != 4)
            {
                throw new OverfitFormatException(
                    $"MaxPool: input rank should be 4, got {inputShape.Length}.");
            }

            int batch = inputShape[0], channels = inputShape[1], h = inputShape[2], w = inputShape[3];

            var outH = (h + 2 * padding - poolSize) / stride + 1;
            var outW = (w + 2 * padding - poolSize) / stride + 1;

            shapes.SetShape(node.Outputs[0], [batch, channels, outH, outW]);

            return new MaxPool2DLayer(channels, h, w, poolSize, stride, padding);
        }
    }
}
