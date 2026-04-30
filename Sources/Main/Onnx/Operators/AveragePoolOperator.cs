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
                : throw new InvalidDataException("AveragePool missing required 'kernel_shape'.");

            var strides = node.Attributes.TryGetValue("strides", out var st)
                ? st.IntArray
                : new long[] { 1, 1 };

            var pads = node.Attributes.TryGetValue("pads", out var p)
                ? p.IntArray
                : new long[] { 0, 0, 0, 0 };

            var countIncludePad = node.Attributes.TryGetValue("count_include_pad", out var cip)
                && cip.IntValue == 1;

            var ceilMode = node.Attributes.TryGetValue("ceil_mode", out var cm)
                ? cm.IntValue : 0L;

            var dilations = node.Attributes.TryGetValue("dilations", out var dil)
                ? dil.IntArray : new long[] { 1, 1 };

            var autoPad = node.Attributes.TryGetValue("auto_pad", out var ap)
                ? ap.StringValue : "NOTSET";

            if (kernelShape.Length != 2 || kernelShape[0] != kernelShape[1])
            {
                throw new NotSupportedException(
                    $"AveragePool: only square kernels supported, got [{string.Join(",", kernelShape)}].");
            }

            if (strides[0] != strides[1])
            {
                throw new NotSupportedException(
                    $"AveragePool: non-square strides [{strides[0]},{strides[1]}] not supported.");
            }

            if (pads.Length == 4 && (pads[0] != pads[2] || pads[1] != pads[3]))
            {
                throw new NotSupportedException(
                    $"AveragePool: asymmetric padding [{string.Join(",", pads)}] not supported.");
            }

            if (ceilMode != 0)
            {
                throw new NotSupportedException("AveragePool: ceil_mode=1 not supported.");
            }

            if (dilations.Any(d => d != 1))
            {
                throw new NotSupportedException(
                    $"AveragePool: dilations [{string.Join(",", dilations)}] not supported.");
            }

            if (!string.IsNullOrEmpty(autoPad) && autoPad != "NOTSET")
            {
                throw new NotSupportedException(
                    $"AveragePool: auto_pad='{autoPad}' not supported. Use explicit pads.");
            }

            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new InvalidDataException(
                    $"AveragePool: input '{node.Inputs[0]}' has no known shape.");

            if (inputShape.Length != 4)
            {
                throw new InvalidDataException(
                    $"AveragePool: expected 4-D [N,C,H,W], got rank {inputShape.Length}.");
            }

            int batch    = inputShape[0];
            int channels = inputShape[1];
            int h        = inputShape[2];
            int w        = inputShape[3];
            int k        = (int)kernelShape[0];
            int stride   = (int)strides[0];
            int padding  = pads.Length >= 1 ? (int)pads[0] : 0;

            int outH = (h + 2 * padding - k) / stride + 1;
            int outW = (w + 2 * padding - k) / stride + 1;
            shapes.SetShape(node.Outputs[0], [batch, channels, outH, outW]);

            return new AveragePool2DLayer(channels, h, w, k, padding, stride, countIncludePad);
        }
    }
}
