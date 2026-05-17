// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        /// <summary>
        /// Fused softmax + cross-entropy loss.
        /// Returns a scalar node [1] containing the mean loss over the batch.
        /// </summary>
        /// <summary>
        /// Fused softmax + cross-entropy loss.
        /// PR5-7c: implementation moved from TensorMath.SoftmaxCrossEntropy.
        /// probsNode stored as GraphAuxiliary — disposed by Reset().
        /// </summary>
        public AutogradNode SoftmaxCrossEntropy(
            AutogradNode logits,
            AutogradNode target)
        {
            int rows = logits.Shape.D0, cols = logits.Shape.D1;
            var total = 0f;

            // probsNode: auxiliary tensor needed by backward, not the primary output.
            var probsNode = CreateAuxiliary(new TensorShape(rows, cols), clearMemory: false);

            var inS = logits.DataView.AsReadOnlySpan();
            var targetS = target.DataView.AsReadOnlySpan();
            var probS = probsNode.DataView.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var pR = probS.Slice(r * cols, cols);
                TensorPrimitives.SoftMax(inS.Slice(r * cols, cols), pR);

                var tR = targetS.Slice(r * cols, cols);
                for (var c = 0; c < cols; c++)
                {
                    if (tR[c] > 0.5f)
                    {
                        total -= MathF.Log(MathF.Max(pR[c], 1e-7f));
                    }
                }
            }

            var output = CreateTemporary(new TensorShape(1), logits.RequiresGrad, clearMemory: false);
            output.DataView.AsSpan()[0] = total / rows;

            if (logits.RequiresGrad)
            {
                Record(OpCode.SoftmaxCrossEntropy, output, logits, target, c0: probsNode, contextCount: 1);
            }
            else
            {
                // probsNode not needed for backward — dispose immediately to free arena slot.
                probsNode.Dispose();
            }

            return output;
        }

        /// <summary>
        /// Mean squared error loss.
        /// Returns a scalar node [1] containing the mean loss over the batch.
        /// </summary>
        public AutogradNode MSELoss(
            AutogradNode prediction,
            AutogradNode target)
        {
            return TensorMath.MSELoss(this, prediction, target);
        }

        /// <summary>
        /// Directional loss — penalises predictions that move away from the target direction.
        /// </summary>
        public AutogradNode DirectionalLoss(
            AutogradNode prediction,
            AutogradNode target,
            float gamma = 10f)
        {
            return TensorMath.DirectionalLoss(this, prediction, target, gamma);
        }
    }
}
