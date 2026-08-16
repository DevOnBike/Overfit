// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>What kind of externally visible thing an <see cref="ApiMember"/> describes.</summary>
    internal enum ApiMemberKind
    {
        /// <summary>A visible type: its accessibility, shape and base type.</summary>
        Type,

        /// <summary>A visible method or constructor, excluding property and event accessors.</summary>
        Method,

        /// <summary>A visible field.</summary>
        Field,

        /// <summary>A visible property. Its accessors are folded in and not reported as methods.</summary>
        Property,

        /// <summary>A visible event. Its accessors are folded in and not reported as methods.</summary>
        Event,
    }
}
