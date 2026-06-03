// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Mp3;

namespace DevOnBike.Overfit.Audio
{
    /// <summary>Pure-C# MPEG-1/2/2.5 Layer III (MP3) decoder.</summary>
    public sealed class Mp3AudioDecoder : IAudioDecoder
    {
        public bool CanRead(string path) => path.EndsWith(".mp3", StringComparison.OrdinalIgnoreCase);

        public float[] ReadMono(Stream stream, out int sampleRate) => Mp3Reader.ReadMono(stream, out sampleRate);
    }
}
