// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Decodes a container/codec into mono 32-bit float PCM in [-1, 1]. Lets consumers (e.g. the Whisper
    /// frontend) stay format-agnostic — new sources (codecs, capture devices) plug in without touching them.
    /// </summary>
    public interface IAudioDecoder
    {
        /// <summary>True if this decoder recognises <paramref name="path"/> (by extension).</summary>
        bool CanRead(string path);

        /// <summary>Decodes <paramref name="stream"/> to mono float PCM, reporting its native sample rate.</summary>
        float[] ReadMono(Stream stream, out int sampleRate);
    }
}
