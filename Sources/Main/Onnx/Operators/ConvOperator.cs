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
    /// Maps ONNX <c>Conv</c> to <see cref="ConvLayer"/>.
    ///
    /// MVP constraints (clear error thrown otherwise):
    ///   - 2D convolution only
    ///   - Square kernel
    ///   - padding = 0 (VALID convolution)
    ///   - stride = 1
    ///   - dilation = 1
    ///   - group = 1 (no depthwise/grouped conv)
    ///
    /// Note: PyTorch eval() mode folds BatchNorm into Conv weights, so no
    /// BatchNormalization operator appears in the exported graph.
    /// </summary>
    internal static class ConvOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var group = node.Attributes.TryGetValue("group", out var g) ? g.IntValue : 1L;
            var dilations = node.Attributes.TryGetValue("dilations", out var d)
                ? d.IntArray
                : new long[] { 1, 1 };
            var strides = node.Attributes.TryGetValue("strides", out var s)
                ? s.IntArray
                : new long[] { 1, 1 };
            var pads = node.Attributes.TryGetValue("pads", out var p)
                ? p.IntArray
                : new long[] { 0, 0, 0, 0 };

            if (group != 1)
            {
                throw new NotSupportedException(
                    $"Conv group={group}: grouped/depthwise convolution not yet supported.");
            }

            if (dilations.Any(dv => dv != 1))
            {
                throw new NotSupportedException(
                    $"Conv dilations=[{string.Join(",", dilations)}]: dilated conv not yet supported.");
            }

            if (strides.Any(sv => sv != 1))
            {
                throw new NotSupportedException(
                    $"Conv strides=[{string.Join(",", strides)}]: stride != 1 not yet supported.");
            }

            if (pads.Any(pv => pv != 0))
            {
                throw new NotSupportedException(
                    $"Conv pads=[{string.Join(",", pads)}]: padding != 0 not yet supported.");
            }

            // Weight tensor: [outC, inC, kH, kW]
            var weightTensor = initializers[node.Inputs[1]];

            if (weightTensor.Dims.Length != 4)
            {
                throw new InvalidDataException(
                    $"Conv weight '{weightTensor.Name}' rank should be 4, got {weightTensor.Dims.Length}.");
            }

            int outC = (int)weightTensor.Dims[0];
            int inC = (int)weightTensor.Dims[1];
            int kH = (int)weightTensor.Dims[2];
            int kW = (int)weightTensor.Dims[3];

            if (kH != kW)
            {
                throw new NotSupportedException(
                    $"Conv: non-square kernels not supported ({kH}×{kW}).");
            }

            var kernelData = OnnxImporter.DecodeFloatTensor(weightTensor);

            // Input shape: [batch, inC, h, w]
            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new InvalidDataException(
                    $"Conv: input '{node.Inputs[0]}' has no known shape in context.");

            if (inputShape.Length != 4)
            {
                throw new InvalidDataException(
                    $"Conv: input rank should be 4, got {inputShape.Length}.");
            }

            int h = inputShape[2], w = inputShape[3];

            var layer = new ConvLayer(inC, outC, h, w, kH);

            // Third input is per-channel bias (optional but common in PyTorch exports)
            if (node.Inputs.Count >= 3 && !string.IsNullOrEmpty(node.Inputs[2])
                && initializers.TryGetValue(node.Inputs[2], out var biasTensor))
            {
                var biasData = OnnxImporter.DecodeFloatTensor(biasTensor);
                layer.LoadParameters(kernelData, biasData);
            }
            else
            {
                layer.LoadParameters(kernelData);
            }

            // VALID convolution output size
            int outH = h - kH + 1;
            int outW = w - kW + 1;
            shapes.SetShape(node.Outputs[0], [inputShape[0], outC, outH, outW]);

            return layer;
        }
    }
}
