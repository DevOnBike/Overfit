// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Tools
{
    /// <summary>The JSON type a tool argument's value is constrained to at decode time.</summary>
    public enum ToolParameterKind : byte
    {
        /// <summary>A JSON string — the value must open with <c>"</c> and is a quoted string.</summary>
        String = 0,

        /// <summary>A JSON number (integer or fractional, RFC 8259).</summary>
        Number,

        /// <summary>A JSON integer — same grammar as <see cref="Number"/> minus the fraction/exponent.</summary>
        Integer,

        /// <summary>A JSON boolean — exactly <c>true</c> or <c>false</c>.</summary>
        Boolean,
    }
}