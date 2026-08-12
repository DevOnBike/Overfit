// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>How a member differs between two builds.</summary>
    internal enum DifferenceKind
    {
        /// <summary>Present on the right and not on the left.</summary>
        Added,

        /// <summary>Present on the left and not on the right.</summary>
        Removed,

        /// <summary>Present on both, but not the same — a changed body, or a changed signature.</summary>
        Changed,
    }
}
