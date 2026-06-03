// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>Container-level metadata probed by walking the MP3 frame headers (no audio decoded).</summary>
    internal readonly record struct Mp3Info(int SampleRate, int Channels, int FrameCount, int SampleCount)
    {
        public double DurationSeconds => SampleRate > 0 ? (double)SampleCount / SampleRate : 0;
    }
}
