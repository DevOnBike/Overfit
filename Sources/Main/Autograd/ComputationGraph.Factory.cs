// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        // ─────────────────────────────────────────────────────────────────────
        // Graph factory methods — Etap 3 of OverfitArchitectureRefactorPlan
        //
        // All node creation goes through these methods so that Ownership is set
        // correctly from birth. This replaces scattered `new AutogradNode(...)`
        // calls in TensorMath with graph-aware, ownership-tagged creation.
        //
        // Current semantics (Etap 3):
        //   - Ownership is set as metadata only.
        //   - Reset() / Dispose() behaviour is unchanged — still uses existing logic.
        //   - Debug-mode assertions will be added in Etap 8 (graph Reset cleanup).
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Creates a graph-owned intermediate activation tensor.
        /// Lives for the duration of one forward pass; disposed by <see cref="Reset"/>.
        /// </summary>
        internal AutogradNode CreateTemporary(
            TensorShape shape,
            bool requiresGrad,
            bool clearMemory = true)
        {
            var storage = AllocateIntermediate(shape.Size);

            if (clearMemory)
            {
                storage.AsSpan().Clear();
            }

            return new AutogradNode(storage, shape, requiresGrad)
            {
                Ownership = AutogradNodeOwnership.GraphTemporary,
            };
        }

        /// <summary>
        /// Creates a graph-owned auxiliary tensor (e.g., MaxPool index map,
        /// BatchNorm saved mean). Disposed by <see cref="Reset"/> alongside temporaries.
        /// </summary>
        internal AutogradNode CreateAuxiliary(
            TensorShape shape,
            bool clearMemory = false)
        {
            var storage = AllocateIntermediate(shape.Size);

            if (clearMemory)
            {
                storage.AsSpan().Clear();
            }

            return new AutogradNode(storage, shape, requiresGrad: false)
            {
                Ownership = AutogradNodeOwnership.GraphAuxiliary,
            };
        }

        /// <summary>
        /// Wraps externally-owned storage in an AutogradNode without taking ownership.
        /// The graph will NOT dispose the storage. Use for preallocated batch buffers
        /// that outlive a single forward pass (e.g., xBNode / yBNode in training loops).
        /// </summary>
        /// <summary>
        /// Wraps externally-owned storage without taking ownership.
        /// Uses <see cref="AutogradNode.CreateBorrowed"/> so that <c>ownsDataStorage = false</c>
        /// is enforced — fixing the bug where the old public constructor always set it to true.
        /// </summary>
        internal static AutogradNode CreateExternalBorrowed(
            TensorStorage<float> storage,
            TensorShape shape,
            bool requiresGrad = false)
        {
            return AutogradNode.CreateBorrowed(storage, shape, requiresGrad);
        }
    }
}
