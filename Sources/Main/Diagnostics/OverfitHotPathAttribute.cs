// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// Marks a method, constructor, property or whole type as a zero-allocation hot path. Inside a
    /// marked member (or any member of a marked type), the in-repo Overfit performance analyzers
    /// (<c>OVERFIT001</c>–<c>OVERFIT014</c>) escalate from their per-directory configured severity to
    /// a hard build <b>error</b> (reported as <c>OVERFIT900</c>) — the per-member evolution of the
    /// per-directory <c>.editorconfig</c> ratchet. Use it to pin a specific decode/inference routine
    /// to the no-hidden-allocations contract regardless of which directory it happens to live in.
    ///
    /// The attribute carries no runtime behaviour; it exists purely as a compile-time marker the
    /// analyzers read from metadata. If a flagged allocation inside a marked member is genuinely
    /// justified, either suppress that specific site with <c>#pragma warning disable OVERFIT900</c>
    /// plus a reason, or drop the attribute — do not silence it globally.
    /// </summary>
    [AttributeUsage(
        AttributeTargets.Method | AttributeTargets.Constructor | AttributeTargets.Property | AttributeTargets.Class | AttributeTargets.Struct,
        Inherited = false,
        AllowMultiple = false)]
    public sealed class OverfitHotPathAttribute : Attribute
    {
    }
}
