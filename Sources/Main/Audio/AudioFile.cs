// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Format-dispatching audio reader. Resolves a path to the right <see cref="IAudioDecoder"/> by extension
    /// (WAV and MP3 are built in) and decodes to mono float PCM. Callers can also supply their own decoders.
    /// </summary>
    public static class AudioFile
    {
        private static readonly IAudioDecoder[] Default = { new WavAudioDecoder(), new Mp3AudioDecoder() };

        /// <summary>The built-in decoders, in resolution order (WAV, MP3).</summary>
        public static IReadOnlyList<IAudioDecoder> Decoders => Default;

        /// <summary>True if a built-in decoder handles <paramref name="path"/>.</summary>
        public static bool IsSupported(string path)
        {
            foreach (var d in Default)
            {
                if (d.CanRead(path)) { return true; }
            }
            return false;
        }

        /// <summary>Decodes <paramref name="path"/> to mono float PCM, choosing a decoder by extension.</summary>
        public static float[] ReadMono(string path, out int sampleRate)
        {
            foreach (var d in Default)
            {
                if (d.CanRead(path))
                {
                    using var fs = File.OpenRead(path);
                    return d.ReadMono(fs, out sampleRate);
                }
            }
            throw new OverfitRuntimeException(
                $"No audio decoder for '{Path.GetExtension(path)}' (supported: .wav, .mp3).");
        }
    }
}
