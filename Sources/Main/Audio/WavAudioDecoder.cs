// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>Pure-C# WAV decoder (16-bit PCM / 32-bit float, stereo downmixed to mono).</summary>
    public sealed class WavAudioDecoder : IAudioDecoder
    {
        public bool CanRead(string path) => path.EndsWith(".wav", StringComparison.OrdinalIgnoreCase);

        public float[] ReadMono(Stream stream, out int sampleRate) => WavReader.ReadMono(stream, out sampleRate);
    }
}
