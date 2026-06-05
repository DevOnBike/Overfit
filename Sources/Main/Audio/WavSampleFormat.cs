// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>Sample encoding for a written WAV file — the two formats <see cref="WavReader"/> reads back.</summary>
    public enum WavSampleFormat
    {
        /// <summary>16-bit signed PCM (audioFormat 1). Smaller; ~16-bit dynamic range.</summary>
        Pcm16,

        /// <summary>32-bit IEEE float (audioFormat 3). Lossless round-trip of the in-memory samples.</summary>
        Float32,
    }
}
