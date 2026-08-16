// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// The release gate's answer, in the three words a person cutting a release has to act on.
    ///
    /// <para><b>Coarser than <see cref="ChangeLevel"/> on purpose.</b> The level says what happened; this says
    /// what to do about it — <see cref="Breaking"/> obliges a decision written down in the plan and a version
    /// bump per <c>CHANGELOG.md</c>'s policy, <see cref="Additive"/> obliges neither.</para>
    ///
    /// <para><b><see cref="Unchanged"/> is about the SURFACE, not about the build.</b> Two builds of the same
    /// public API always differ in IL and in build stamps, and comparing a released package against a
    /// candidate always moves the assembly version. Reading this as "nothing changed" would be wrong in a way
    /// that matters: it means "no consumer sees a difference in what they can call", and a behaviour change
    /// inside an unchanged signature is invisible to it.</para>
    /// </summary>
    internal enum ApiCompatibilityVerdict
    {
        /// <summary>No visible member was added, removed or changed.</summary>
        Unchanged = 0,

        /// <summary>New surface only. Breaks nobody, needs no decision.</summary>
        Additive = 1,

        /// <summary>
        /// At least one finding above <see cref="ApiCompatibilityGate.HighestAllowedLevel"/> — a source break,
        /// a binary break, or a silent behaviour change baked into a consumer's own IL.
        /// </summary>
        Breaking = 2,
    }
}
