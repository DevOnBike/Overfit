// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// How a <c>TensorStorage&lt;T&gt;</c> holds its memory. Three states, which is why a <c>bool</c> was
    /// not enough.
    ///
    /// <para><b>The counter carried one bit where it needed two, and the result was a metric that
    /// misleads rather than one that is merely absent.</b> <c>RecordTensorStorageCreated</c> took
    /// <c>bool borrowed</c>, so the <c>Unpooled</c> path — a GC-owned array, used for exactly the
    /// long-lived things that must <i>not</i> go through the pool, model weights above all — reported
    /// itself as pooled. Anyone watching pool pressure on <c>overfit.tensor_storage.pooled.created</c> was
    /// reading one-time weight-load allocations mixed into a churn signal, and the mix is largest at
    /// startup, which is exactly when someone looks.</para>
    /// </summary>
    public enum TensorStorageKind
    {
        /// <summary>Rented from <c>ArrayPool</c> and returned on dispose. Short-lived scratch.</summary>
        Pooled = 0,

        /// <summary>
        /// A GC-owned exact-sized array, never returned to a pool. Long-lived — model weights — where
        /// pool retention would hold rounded-up buckets alive for the life of the process.
        /// </summary>
        Unpooled = 1,

        /// <summary>A slice of a native arena owned by somebody else. Freed by the arena, not here.</summary>
        Borrowed = 2,
    }
}
