// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.DeepLearning.Abstractions
{
    /// <summary>
    ///     Defines the fundamental contract for all neural network modules, including layers and composite models.
    ///     Supports the training/evaluation lifecycle, parameter management, and serialization.
    /// </summary>
    public interface IModule : IDisposable
    {
        /// <summary>
        ///     Gets a value indicating whether the module is currently in training mode.
        ///     This state affects layers like <c>Dropout</c> or <c>BatchNorm</c>.
        /// </summary>
        bool IsTraining { get; }

        /// <summary>
        ///     Sets the module (and its sub-modules) to training mode.
        /// </summary>
        void Train();

        /// <summary>
        ///     Sets the module (and its sub-modules) to evaluation (inference) mode.
        /// </summary>
        void Eval();

        /// <summary>
        ///     Performs the forward pass through the module.
        /// </summary>
        /// <param name="graph">The computation graph to record operations for Autograd. Can be <c>null</c> for inference.</param>
        /// <param name="input">The input tensor node.</param>
        /// <returns>The resulting output tensor node.</returns>
        AutogradNode Forward(ComputationGraph graph, AutogradNode input);

        /// <summary>
        ///     Extremely fast, 0-allocation forward pass for inference (Batch = 1).
        /// </summary>
        void ForwardInference(ReadOnlySpan<float> input, Span<float> output);

        /// <summary>
        ///     Retrieves all learnable parameters (weights and biases) within this module and its children.
        ///     Typically used by the <c>Optimizer</c> to update weights during backpropagation.
        /// </summary>
        IEnumerable<AutogradNode> Parameters();

        /// <summary>
        ///     Serializes the module's parameters to a binary stream.
        /// </summary>
        void Save(BinaryWriter bw);

        /// <summary>
        ///     Deserializes the module's parameters from a binary stream.
        /// </summary>
        void Load(BinaryReader br);

        /// <summary>
        ///     Signals that this module's parameters have been mutated externally (for example, by
        ///     <c>IParameterVectorAdapter.ReadFromVector</c> in the evolutionary pipeline) and any
        ///     cached state derived from those parameters must be invalidated.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///         Default implementation is a no-op. Modules that maintain caches derived from
        ///         <see cref="Parameters"/> (for example, <c>LinearLayer</c>'s transposed-weight
        ///         buffer used by <see cref="ForwardInference"/>) must override this method and
        ///         invalidate those caches lazily — without allocating — so the next inference
        ///         call rebuilds them on demand.
        ///     </para>
        ///     <para>
        ///         Composite modules (<c>Sequential</c>, <c>ResidualBlock</c>, <c>LstmAutoencoder</c>)
        ///         must propagate the call to all child modules.
        ///     </para>
        ///     <para>
        ///         <see cref="Load(BinaryReader)"/> and <see cref="Train"/> are NOT required to
        ///         call this method — they already invalidate their own caches through their
        ///         existing protocols.
        ///     </para>
        /// </remarks>
        void InvalidateParameterCaches();
    }
}