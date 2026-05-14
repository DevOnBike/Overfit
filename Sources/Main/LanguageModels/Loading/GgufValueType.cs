// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>GGUF metadata value type tags.</summary>
    public enum GgufValueType : uint
    {
        UInt8   = 0,
        Int8    = 1,
        UInt16  = 2,
        Int16   = 3,
        UInt32  = 4,
        Int32   = 5,
        Float32 = 6,
        Bool    = 7,
        String  = 8,
        Array   = 9,
        UInt64  = 10,
        Int64   = 11,
        Float64 = 12,
    }
}