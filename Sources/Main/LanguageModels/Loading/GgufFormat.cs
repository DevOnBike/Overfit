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

}
