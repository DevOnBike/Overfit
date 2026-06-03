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

    /// <summary>
    /// Pure-C# MPEG-1/2/2.5 Layer III (MP3) reader — no native binaries, no external libraries, no Python.
    /// <see cref="ReadMono"/> decodes to mono 32-bit float PCM (the shape <see cref="WavReader"/> returns), so
    /// the Whisper frontend can consume <c>.mp3</c> directly. Frame-walking + header parsing live here; the
    /// per-frame Layer III synthesis is in <see cref="Mp3Decoder"/>.
    /// </summary>
    public static class Mp3Reader
    {
        /// <summary>Reads an MP3 file and returns mono 32-bit float samples in [-1, 1].</summary>
        public static float[] ReadMono(string path, out int sampleRate)
        {
            using var fs = File.OpenRead(path);
            return ReadMono(fs, out sampleRate);
        }

        /// <summary>Reads an MP3 stream and returns mono 32-bit float samples in [-1, 1].</summary>
        public static float[] ReadMono(Stream stream, out int sampleRate)
        {
            var bytes = ReadAll(stream);
            var decoder = new Mp3Decoder();
            return decoder.DecodeMono(bytes, out sampleRate);
        }

        /// <summary>
        /// Walks the frame headers only and reports sample rate / channels / frame &amp; sample counts — a cheap
        /// container check (does not decode audio). Useful as a first validation layer.
        /// </summary>
        internal static Mp3Info Probe(byte[] bytes)
        {
            var pos = SkipId3(bytes);
            var sampleRate = 0;
            var channels = 0;
            var frames = 0;
            var samples = 0;

            while (pos + 4 <= bytes.Length)
            {
                if (!Mp3FrameHeader.TryParse(bytes, pos, out var h))
                {
                    pos++; // resync byte-by-byte
                    continue;
                }

                var len = h.FrameLengthBytes;
                if (len < 4 || pos + len > bytes.Length)
                {
                    break;
                }

                if (frames == 0)
                {
                    sampleRate = h.SampleRate;
                    channels = h.Channels;
                }

                frames++;
                samples += h.SamplesPerFrame;
                pos += len;
            }

            return new Mp3Info(sampleRate, channels, frames, samples);
        }

        /// <summary>Skips an ID3v2 tag if present (10-byte header + syncsafe size), returning the audio offset.</summary>
        internal static int SkipId3(byte[] bytes)
        {
            if (bytes.Length >= 10 && bytes[0] == (byte)'I' && bytes[1] == (byte)'D' && bytes[2] == (byte)'3')
            {
                // size is a 28-bit syncsafe integer in bytes[6..9]
                var size = (bytes[6] << 21) | (bytes[7] << 14) | (bytes[8] << 7) | bytes[9];
                return 10 + size;
            }
            return 0;
        }

        private static byte[] ReadAll(Stream stream)
        {
            if (stream is MemoryStream ms)
            {
                return ms.ToArray();
            }
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            return buffer.ToArray();
        }
    }
}
