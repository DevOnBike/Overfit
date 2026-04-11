// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    ///     Implements the Stochastic Gradient Descent (SGD) optimizer.
    ///     Utilizes hardware-accelerated parameter updates via SIMD primitives.
    /// </summary>
    public sealed class SGD : IOptimizer
    {
        private readonly AutogradNode[] _parameters;

        /// <param name="parameters">
        ///     Collection of parameters to be optimized. Only nodes with <c>RequiresGrad=true</c> are
        ///     tracked.
        /// </param>
        /// <param name="learningRate">The initial learning rate for weight updates.</param>
        public SGD(IEnumerable<AutogradNode> parameters, float learningRate)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);

            var paramList = new List<AutogradNode>();

            foreach (var p in parameters)
            {
                if (p.RequiresGrad)
                {
                    paramList.Add(p);
                }
            }

            _parameters = [.. paramList];
            LearningRate = learningRate;
        }

        public float LearningRate { get; set; }

        /// <summary>
        ///     Performs a single optimization step.
        ///     Applies the update rule: w = w - lr * grad.
        /// </summary>
        public void Step()
        {
            var negativeLr = -LearningRate;

            foreach (var p in _parameters)
            {
                TensorPrimitives.MultiplyAdd(
                p.Grad.AsSpan(),
                negativeLr,
                p.Data.AsSpan(),
                p.Data.AsSpan());
            }
        }

        /// <summary>
        ///     Resets the gradients of all managed parameters to zero.
        ///     Should be called before each forward pass.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var p in _parameters)
            {
                p.Grad.AsSpan().Clear();
            }
        }
    }
}