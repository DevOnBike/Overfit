// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        /// <summary>
        /// Rectified Linear Unit: output[i] = max(0, input[i]).
        /// </summary>
        public AutogradNode Relu(AutogradNode input)
        {
            return TensorMath.ReLU(this, input);
        }

        /// <summary>
        /// Sigmoid activation: output[i] = 1 / (1 + exp(-input[i])).
        /// </summary>
        public AutogradNode Sigmoid(AutogradNode input)
        {
            return TensorMath.Sigmoid(this, input);
        }

        /// <summary>
        /// Hyperbolic tangent activation: output[i] = tanh(input[i]).
        /// </summary>
        public AutogradNode Tanh(AutogradNode input)
        {
            return TensorMath.Tanh(this, input);
        }

        /// <summary>
        /// Dropout regularisation (identity at inference when isTraining=false).
        /// </summary>
        public AutogradNode Dropout(AutogradNode input, float probability, bool isTraining)
        {
            return TensorMath.Dropout(this, input, probability, isTraining);
        }

        /// <summary>
        /// Batch Normalisation (1-D, over feature dimension).
        /// </summary>
        public AutogradNode BatchNorm1D(
            AutogradNode input,
            AutogradNode gamma,
            AutogradNode beta,
            TensorStorage<float> runningMean,
            TensorStorage<float> runningVar,
            float momentum,
            float eps,
            bool isTraining)
        {
            return TensorMath.BatchNorm1D(
                this,
                input,
                gamma,
                beta,
                runningMean,
                runningVar,
                momentum,
                eps,
                isTraining);
        }
    }
}
