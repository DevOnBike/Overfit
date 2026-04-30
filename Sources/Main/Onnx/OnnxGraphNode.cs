// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Represents one execution step in an ONNX DAG inference plan.
    ///
    /// Each node wraps an <see cref="IModule"/> and records:
    ///   - Which buffer indices to read as inputs (InputSlots).
    ///   - Which buffer index to write the output to (OutputSlot).
    ///   - The element size of the output (for buffer allocation).
    ///
    /// Buffers are pre-allocated float arrays indexed by slot. Slot 0 is always
    /// the model input. The model output is always the last written slot.
    /// </summary>
    internal sealed class OnnxGraphNode
    {
        public OnnxGraphNode(
            IModule module,
            int[] inputSlots,
            int outputSlot,
            int outputSize)
        {
            Module      = module;
            InputSlots  = inputSlots;
            OutputSlot  = outputSlot;
            OutputSize  = outputSize;
        }

        /// <summary>The layer that computes this node's output.</summary>
        public IModule Module { get; }

        /// <summary>
        /// Indices into the buffer array for this node's inputs.
        /// Length 1 for regular layers. Length 2 for Add (skip connection).
        /// </summary>
        public int[] InputSlots { get; }

        /// <summary>Index into the buffer array where this node writes its output.</summary>
        public int OutputSlot { get; }

        /// <summary>Number of floats in this node's output (for buffer sizing).</summary>
        public int OutputSize { get; }
    }
}
