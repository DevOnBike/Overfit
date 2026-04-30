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
    /// Maps ONNX <c>BatchNormalization</c> to <see cref="BatchNorm1D"/>.
    ///
    /// ONNX BatchNormalization inputs:
    ///   0  X           — input tensor [N, C, *]
    ///   1  scale       — per-channel gamma (learnable)
    ///   2  B           — per-channel beta  (learnable)
    ///   3  input_mean  — running mean  (updated during training, read during inference)
    ///   4  input_var   — running variance
    ///
    /// Attributes:
    ///   epsilon       (float, default 1e-5)
    ///   momentum      (float, default 0.9)  — ONNX convention: used as keep factor for running stats
    ///   training_mode (int,   default 0)    — 0 = eval, 1 = train
    ///
    /// Note: PyTorch eval() export folds BatchNorm weights into Conv — this operator only
    /// appears in train-mode exports or models that explicitly keep BN separate.
    ///
    /// Shape: output has the same shape as input.
    /// </summary>
    internal static class BatchNormOperator
    {
        public static IModule Build(
            OnnxNode node,
            Dictionary<string, OnnxTensor> initializers,
            OnnxShapeContext shapes)
        {
            // ── Attributes ────────────────────────────────────────────────────
            var eps = node.Attributes.TryGetValue("epsilon", out var epsAttr)
                ? (float)epsAttr.FloatValue
                : 1e-5f;

            var onnxMomentum = node.Attributes.TryGetValue("momentum", out var momAttr)
                ? (float)momAttr.FloatValue
                : 0.9f;

            var trainingMode = node.Attributes.TryGetValue("training_mode", out var tmAttr)
                ? tmAttr.IntValue
                : 0L;

            if (trainingMode != 0)
            {
                throw new NotSupportedException(
                    "BatchNormalization: training_mode=1 is not supported in ONNX import. " +
                    "Export your model with model.eval() to use inference-mode BatchNorm, " +
                    "or use PyTorch eval() export which folds BN into Conv weights.");
            }

            // ── Initializers (weights) ────────────────────────────────────────
            var scaleTensor = initializers[node.Inputs[1]];
            var biasTensor  = initializers[node.Inputs[2]];
            var meanTensor  = initializers[node.Inputs[3]];
            var varTensor   = initializers[node.Inputs[4]];

            var scaleData = OnnxImporter.DecodeFloatTensor(scaleTensor);
            var biasData  = OnnxImporter.DecodeFloatTensor(biasTensor);
            var meanData  = OnnxImporter.DecodeFloatTensor(meanTensor);
            var varData   = OnnxImporter.DecodeFloatTensor(varTensor);

            var numFeatures = scaleData.Length;

            // ── Input shape ───────────────────────────────────────────────────
            var inputShape = shapes.GetShape(node.Inputs[0])
                ?? throw new InvalidDataException(
                    $"BatchNormalization: input '{node.Inputs[0]}' has no known shape.");

            // ── Build layer ───────────────────────────────────────────────────
            var layer = new BatchNorm1D(numFeatures)
            {
                Eps = eps,
                // ONNX momentum is the keep factor for running stats.
                // Overfit momentum is the update factor: update = momentum * new + (1-momentum) * old.
                // Conversion: overfit_momentum = 1 - onnx_momentum.
                Momentum = 1f - onnxMomentum,
            };

            // Load learnable parameters
            layer.Gamma.LoadData(scaleData);
            layer.Beta.LoadData(biasData);

            // Load running statistics from ONNX initializers.
            // In eval mode, these are used directly (not updated).
            meanData.CopyTo(layer.RunningMean.AsSpan());
            varData.CopyTo(layer.RunningVar.AsSpan());

            // Switch to eval mode — uses running stats for inference.
            layer.Eval();

            // ── Shape propagation ─────────────────────────────────────────────
            // BatchNorm is shape-preserving.
            shapes.SetShape(node.Outputs[0], inputShape);

            return layer;
        }
    }
}
