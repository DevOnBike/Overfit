// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        /// <summary>
        /// Fused softmax + cross-entropy loss.
        /// Returns a scalar node [1] containing the mean loss over the batch.
        /// </summary>
        public AutogradNode SoftmaxCrossEntropy(
            AutogradNode logits,
            AutogradNode target)
        {
            return TensorMath.SoftmaxCrossEntropy(this, logits, target);
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
