// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    ///     Defines the fundamental contract for all weight optimization algorithms (e.g., SGD, Adam).
    ///     Manages the parameter update logic based on gradients computed during backpropagation.
    /// </summary>
    public interface IOptimizer
    {
        /// <summary>
        ///     Gets or sets the learning rate, determining the step size taken towards the minimum of the loss function.
        /// </summary>
        float LearningRate { get; set; }

        /// <summary>
        ///     Performs a single optimization step.
        ///     Updates the learnable parameters (weights and biases) using their calculated gradients.
        /// </summary>
        void Step();

        /// <summary>
        ///     Resets the gradients of all managed parameters to zero.
        ///     This must be called before the start of a new training batch to prevent gradient accumulation.
        /// </summary>
        void ZeroGrad();
    }
}