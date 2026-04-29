// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// Describes the lifecycle and ownership semantics of an <see cref="AutogradNode"/>.
    ///
    /// Used by <see cref="ComputationGraph.Reset"/> to determine which nodes should be
    /// disposed (graph-owned temporaries) vs preserved (model parameters, external inputs).
    ///
    /// In the current implementation this is metadata only — Reset/Dispose behaviour
    /// has not changed yet. The enum enables auditing ownership via debug assertions
    /// and prepares for Etap 8 (graph Reset cleanup by ownership).
    /// </summary>
    public enum AutogradNodeOwnership
    {
        /// <summary>
        /// Ownership not yet classified. Default for nodes created before this enum existed.
        /// Treated conservatively — Reset does not dispose Unknown nodes.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// Long-lived trainable model state. Lives for the lifetime of the layer/model.
        /// Must NOT be disposed by graph.Reset().
        /// Examples: LinearLayer.Weights, ConvLayer.Kernels, BatchNorm gamma/beta.
        /// </summary>
        Parameter = 1,

        /// <summary>
        /// Intermediate activation created during one forward pass.
        /// Owned by the graph; safe to dispose (and free arena storage) after graph.Reset().
        /// Examples: ReLU output, Conv output, hidden layer activations.
        /// </summary>
        GraphTemporary = 2,

        /// <summary>
        /// Auxiliary tensor needed by backward but not part of the primary activation chain.
        /// Owned by the graph; disposed by graph.Reset().
        /// Examples: MaxPool index map, BatchNorm saved mean/variance.
        /// </summary>
        GraphAuxiliary = 3,

        /// <summary>
        /// Node wrapping externally-owned storage (e.g., preallocated batch input buffers).
        /// The graph does NOT own the underlying storage; Reset must not dispose it.
        /// Examples: xBNode / yBNode in MnistTrainingTests (reused across batches).
        /// </summary>
        ExternalBorrowed = 4,

        /// <summary>
        /// A lightweight view over another node's storage (no independent storage allocation).
        /// Must NOT be disposed independently — the owner manages the storage lifetime.
        /// Examples: FlattenLayer output (view of MaxPool output with different shape).
        /// </summary>
        View = 5,
    }
}
