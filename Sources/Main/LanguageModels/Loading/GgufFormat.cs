// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// GGUF file format constants. Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    /// </summary>
    public static class GgufFormat
    {
        /// <summary>"GGUF" magic as little-endian uint32.</summary>
        public const uint Magic = 0x46554747u;

        public const uint SupportedVersionMin = 2;
        public const uint SupportedVersionMax = 3;

        /// <summary>Data section starts at next 32-byte aligned position after tensor info.</summary>
        public const int DataAlignment = 32;
    }

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

    /// <summary>
    /// GGML tensor element types. Etap A supports F32 and F16 only.
    /// Quantized types (Q4_K_M, Q8_0 etc.) will be added in Etap B.
    /// </summary>
    public enum GgmlType : uint
    {
        F32  = 0,
        F16  = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        BF16 = 30,
    }
}
