// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Constraints.Schema
{
    /// <summary>
    /// Allowed JSON-Schema value types at a position, as flags so a <c>"type": ["string","null"]</c>
    /// union (or an absent <c>type</c>, which permits all) is one value. The MVP enforces type
    /// restriction, required/optional object properties, <c>additionalProperties:false</c>, string enums,
    /// nested objects and simple arrays — <c>anyOf</c> / <c>const</c> / min-max-items are out of scope.
    /// </summary>
    [System.Flags]
    public enum JsonSchemaType : byte
    {
        None = 0,
        Object = 1 << 0,
        Array = 1 << 1,
        String = 1 << 2,
        Number = 1 << 3,
        Integer = 1 << 4,
        Boolean = 1 << 5,
        Null = 1 << 6,

        /// <summary>Every type — used for an unconstrained position (no <c>type</c> keyword).</summary>
        Any = Object | Array | String | Number | Integer | Boolean | Null,
    }
}
