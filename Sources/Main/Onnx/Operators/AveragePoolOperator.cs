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
    /// Maps ONNX <c>AveragePool</c> to <see cref="AveragePool2DLayer"/>.
    ///
    /// Supported: square kernel, square stride, symmetric padding,
    /// count_include_pad. Throws clearly for ceil_mode, dilations, auto_pad.
    /// </summary>
    internal static class AveragePoolOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var kernelShape = node.Attributes.TryGetValue("kernel_shape", out var ks)
                ? ks.IntArray
                : throw new OverfitFormatException("AveragePool missing required 'kernel_shape'.");

            var strides = node.Attributes.TryGetValue("strides", out var st)
                ? st.IntArray
                : [1, 1];

            var pads = node.Attributes.TryGetValue("pads", out var p)
                ? p.IntArray
                : [0, 0, 0, 0];

            var countIncludePad = node.Attributes.TryGetValue("count_include_pad", out var cip)
                && cip.IntValue == 1;

            var ceilMode = node.Attributes.TryGetValue("ceil_mode", out var cm)
                ? cm.IntValue : 0L;

            var dilations = node.Attributes.TryGetValue("dilations", out var dil)
                ? dil.IntArray : [1, 1];

            var autoPad = node.Attributes.TryGetValue("auto_pad", out var ap)
                ? ap.StringValue : "NOTSET";

            if (kernelShape.Length != 2 || kernelShape[0] != kernelShape[1])
            {
                throw new OverfitRuntimeException(
                    $"AveragePool: only square kernels supported, got [{string.Join(",", kernelShape)}].");
            }

            if (strides[0] != strides[1])
            {
                throw new OverfitRuntimeException(
                    $"AveragePool: non-square strides [{strides[0]},{strides[1]}] not supported.");
            }

            if (pads.Length == 4 && (pads[0] != pads[2] || pads[1] != pads[3]))
            {
                throw new OverfitRuntimeException(
                    $"AveragePool: asymmetric padding [{string.Join(",", pads)}] not supported.");
            }

            if (ceilMode != 0)
            {
                throw new OverfitRuntimeException("AveragePool: ceil_mode=1 not supported.");
            }

            if (dilations.Any(d => d != 1))
            {
                throw new OverfitRuntimeException(
                    $"AveragePool: dilations [{string.Join(",", dilations)}] not supported.");
            }

            if (!string.IsNullOrEmpty(autoPad) && autoPad != "NOTSET")
            {
                throw new OverfitRuntimeException(
                    $"AveragePool: auto_pad='{autoPad}' not supported. Use explicit pads.");
            }

            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new OverfitFormatException(
                    $"AveragePool: input '{node.Inputs[0]}' has no known shape.");

            if (inputShape.Length != 4)
            {
                throw new OverfitFormatException(
                    $"AveragePool: expected 4-D [N,C,H,W], got rank {inputShape.Length}.");
            }

            var batch = inputShape[0];
            var channels = inputShape[1];
            var h = inputShape[2];
            var w = inputShape[3];
            var k = (int)kernelShape[0];
            var stride = (int)strides[0];
            var padding = pads.Length >= 1 ? (int)pads[0] : 0;

            var outH = (h + 2 * padding - k) / stride + 1;
            var outW = (w + 2 * padding - k) / stride + 1;
            shapes.SetShape(node.Outputs[0], [batch, channels, outH, outW]);

            return new AveragePool2DLayer(channels, h, w, k, padding, stride, countIncludePad);
        }
    }
}
