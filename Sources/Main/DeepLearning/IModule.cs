// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Defines the fundamental contract for all neural network modules, including layers and composite models.
    /// Supports the training/evaluation lifecycle, parameter management, and serialization.
    /// </summary>
    public interface IModule : IDisposable
    {
        /// <summary>
        /// Gets a value indicating whether the module is currently in training mode.
        /// This state affects layers like <c>Dropout</c> or <c>BatchNorm</c>.
        /// </summary>
        bool IsTraining { get; }

        /// <summary>
        /// Sets the module (and its sub-modules) to training mode.
        /// </summary>
        void Train();

        /// <summary>
        /// Sets the module (and its sub-modules) to evaluation (inference) mode.
        /// </summary>
        void Eval();

        /// <summary>
        /// Performs the forward pass through the module.
        /// </summary>
        /// <param name="graph">The computation graph to record operations for Autograd. Can be <c>null</c> for inference.</param>
        /// <param name="input">The input tensor node.</param>
        /// <returns>The resulting output tensor node.</returns>
        AutogradNode Forward(ComputationGraph graph, AutogradNode input);

        /// <summary>
        /// Retrieves all learnable parameters (weights and biases) within this module and its children.
        /// Typically used by the <c>Optimizer</c> to update weights during backpropagation.
        /// </summary>
        IEnumerable<AutogradNode> Parameters();

        /// <summary>
        /// Serializes the module's parameters to a binary stream.
        /// </summary>
        void Save(BinaryWriter bw);

        /// <summary>
        /// Deserializes the module's parameters from a binary stream.
        /// </summary>
        void Load(BinaryReader br);
    }
}