// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Optimizers.Abstractions;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    /// Implements the Stochastic Gradient Descent (SGD) optimizer.
    /// </summary>
    public sealed class SGD : IOptimizer
    {
        private readonly AutogradNode[] _parameters;

        /// <summary>
        /// Initializes a new SGD optimizer.
        /// Only parameters with RequiresGrad=true are tracked.
        /// </summary>
        public SGD(
            IEnumerable<AutogradNode> parameters,
            float learningRate)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);

            var paramList = new List<AutogradNode>();

            foreach (var parameter in parameters)
            {
                if (parameter.RequiresGrad)
                {
                    paramList.Add(parameter);
                }
            }

            _parameters = paramList.ToArray();
            LearningRate = learningRate;
        }

        /// <summary>
        /// Learning rate used by the optimizer.
        /// </summary>
        public float LearningRate { get; set; }

        /// <summary>
        /// Performs a single optimization step.
        /// Applies the update rule:
        ///
        ///     w = w - learningRate * grad
        ///
        /// Implemented as:
        ///
        ///     weights = grad * (-learningRate) + weights
        ///
        /// The same weights buffer is used as both addend and destination.
        /// This requires ElementwiseKernels.MultiplyAdd to support:
        ///
        ///     addend == destination
        ///
        /// Keep the aliasing test before changing this method.
        /// </summary>
        public void Step()
        {
            var negativeLearningRate = -LearningRate;

            for (var i = 0; i < _parameters.Length; i++)
            {
                var parameter = _parameters[i];

                ElementwiseKernels.MultiplyAdd(
                    parameter.GradView.AsReadOnlySpan(),
                    negativeLearningRate,
                    parameter.DataView.AsReadOnlySpan(),
                    parameter.DataView.AsSpan());
            }
        }

        /// <summary>
        /// Resets the gradients of all managed parameters to zero.
        /// Should be called before each forward pass.
        /// </summary>
        public void ZeroGrad()
        {
            for (var i = 0; i < _parameters.Length; i++)
            {
                _parameters[i].ZeroGrad();
            }
        }
    }
}