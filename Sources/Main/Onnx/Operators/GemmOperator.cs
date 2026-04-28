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
    /// Maps ONNX <c>Gemm</c> to <see cref="LinearLayer"/>.
    ///
    /// PyTorch exports nn.Linear as:
    ///   Gemm(input, weight, bias, transA=0, transB=1, alpha=1, beta=1)
    /// where weight shape is [outFeatures, inFeatures].
    ///
    /// Overfit LinearLayer stores weights as [inFeatures, outFeatures], so we
    /// transpose once at import time — zero cost at inference.
    /// </summary>
    internal static class GemmOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            var transA = GetInt(node, "transA", 0);
            var transB = GetInt(node, "transB", 0);
            var alpha = GetFloat(node, "alpha", 1f);
            var beta = GetFloat(node, "beta", 1f);

            if (transA != 0)
            {
                throw new NotSupportedException("Gemm transA=1 not supported.");
            }

            if (alpha != 1f || beta != 1f)
            {
                throw new NotSupportedException(
                    $"Gemm alpha={alpha} beta={beta}: only alpha=1 beta=1 supported.");
            }

            var weightTensor = initializers[node.Inputs[1]];

            if (weightTensor.Dims.Length != 2)
            {
                throw new InvalidDataException(
                    $"Gemm weight '{weightTensor.Name}' rank should be 2, got {weightTensor.Dims.Length}.");
            }

            // transB=1 → weight is [out, in], must transpose to Overfit's [in, out]
            // transB=0 → weight is already [in, out]
            int inFeatures, outFeatures;
            if (transB == 1)
            {
                outFeatures = (int)weightTensor.Dims[0];
                inFeatures = (int)weightTensor.Dims[1];
            }
            else
            {
                inFeatures = (int)weightTensor.Dims[0];
                outFeatures = (int)weightTensor.Dims[1];
            }

            var weightData = OnnxImporter.DecodeFloatTensor(weightTensor);

            // Transpose [out, in] → [in, out] if needed
            if (transB == 1)
            {
                weightData = Transpose2D(weightData, outFeatures, inFeatures);
            }

            // Optional bias (third Gemm input)
            float[]? biasData = null;
            if (node.Inputs.Count >= 3 && !string.IsNullOrEmpty(node.Inputs[2]))
            {
                biasData = OnnxImporter.DecodeFloatTensor(initializers[node.Inputs[2]]);
            }

            var layer = new LinearLayer(inFeatures, outFeatures);
            layer.LoadParameters(weightData, biasData ?? new float[outFeatures]);

            // Propagate output shape
            var inputShape = shapes.GetShape(node.Inputs[0]);
            var batch = inputShape != null ? inputShape[0] : 1;
            shapes.SetShape(node.Outputs[0], [batch, outFeatures]);

            return layer;
        }

        private static float[] Transpose2D(float[] src, int rows, int cols)
        {
            // src layout: [rows, cols] → dst layout: [cols, rows]
            var dst = new float[src.Length];
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++)
                {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
            return dst;
        }

        private static long GetInt(OnnxNode node, string name, long defaultValue)
        {
            return node.Attributes.TryGetValue(name, out var attr) ? attr.IntValue : defaultValue;
        }

        private static float GetFloat(OnnxNode node, string name, float defaultValue)
        {
            return node.Attributes.TryGetValue(name, out var attr) ? attr.FloatValue : defaultValue;
        }
    }
}
