// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>MPEG audio version field (header bits 19–20).</summary>
    internal enum MpegVersion
    {
        Mpeg25 = 0, // MPEG 2.5 (unofficial extension for 8/11.025/12 kHz)
        Reserved = 1,
        Mpeg2 = 2, // MPEG 2 (LSF) 16/22.05/24 kHz
        Mpeg1 = 3, // MPEG 1 32/44.1/48 kHz
    }
}
