
// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.IO;

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Minimal, dependency-free WAV reader for the speech front-end: parses a RIFF/WAVE file and returns
    /// mono float samples in [−1, 1]. Supports 16-bit PCM and 32-bit IEEE float; multi-channel audio is
    /// down-mixed to mono by averaging. No resampling — the caller should feed 16 kHz audio (Whisper's rate).
    /// </summary>
    public static class WavReader
    {
        /// <summary>Reads <paramref name="path"/> → mono float samples; <paramref name="sampleRate"/> receives the rate (Hz).</summary>
        public static float[] ReadMono(string path, out int sampleRate)
        {
            using var fs = File.OpenRead(path);
            return ReadMono(fs, out sampleRate);
        }

        /// <summary>Reads a WAV stream → mono float samples.</summary>
        public static float[] ReadMono(Stream stream, out int sampleRate)
        {
            using var br = new BinaryReader(stream);
            if (ReadTag(br) != "RIFF") { throw new OverfitFormatException("Not a RIFF file."); }
            br.ReadInt32(); // file size
            if (ReadTag(br) != "WAVE") { throw new OverfitFormatException("Not a WAVE file."); }

            int channels = 0, bitsPerSample = 0, audioFormat = 0;
            sampleRate = 0;
            byte[]? data = null;

            while (stream.Position < stream.Length)
            {
                var chunkId = ReadTag(br);
                var chunkSize = br.ReadInt32();
                if (chunkId == "fmt ")
                {
                    audioFormat = br.ReadInt16();   // 1 = PCM, 3 = IEEE float
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32();                 // byte rate
                    br.ReadInt16();                 // block align
                    bitsPerSample = br.ReadInt16();
                    var consumed = 16;
                    if (chunkSize > consumed) { br.ReadBytes(chunkSize - consumed); } // skip extension
                }
                else if (chunkId == "data")
                {
                    data = br.ReadBytes(chunkSize);
                }
                else
                {
                    br.ReadBytes(chunkSize);        // skip unknown chunk
                    if ((chunkSize & 1) == 1) { br.ReadByte(); } // chunks are word-aligned
                }
            }

            if (data is null || channels == 0) { throw new OverfitFormatException("WAV missing fmt/data chunk."); }

            return Decode(data, audioFormat, channels, bitsPerSample);
        }

        // OVERFIT001: by-contract — decodes the WAV payload into a fresh PCM array the caller owns (one
        // allocation per file load, not a per-frame hot path); the down-mix buffer is likewise the output.
#pragma warning disable OVERFIT001
        private static float[] Decode(byte[] data, int audioFormat, int channels, int bitsPerSample)
        {
            var span = data.AsSpan();
            float[] interleaved;
            if (audioFormat == 1 && bitsPerSample == 16)
            {
                var count = data.Length / 2;
                interleaved = new float[count];
                for (var i = 0; i < count; i++)
                {
                    interleaved[i] = BinaryPrimitives.ReadInt16LittleEndian(span.Slice(i * 2, 2)) / 32768f;
                }
            }
            else if (audioFormat == 3 && bitsPerSample == 32)
            {
                var count = data.Length / 4;
                interleaved = new float[count];
                for (var i = 0; i < count; i++)
                {
                    interleaved[i] = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(i * 4, 4));
                }
            }
            else
            {
                throw new OverfitRuntimeException($"Unsupported WAV format (audioFormat={audioFormat}, bits={bitsPerSample}). Use 16-bit PCM or 32-bit float.");
            }

            if (channels == 1) { return interleaved; }

            // Down-mix to mono by averaging channels.
            var frames = interleaved.Length / channels;
            var mono = new float[frames];
            for (var i = 0; i < frames; i++)
            {
                var acc = 0f;
                for (var c = 0; c < channels; c++) { acc += interleaved[i * channels + c]; }
                mono[i] = acc / channels;
            }
            return mono;
        }
#pragma warning restore OVERFIT001

        private static string ReadTag(BinaryReader br)
        {
            Span<byte> tag = stackalloc byte[4];
            if (br.Read(tag) != 4) { throw new EndOfStreamException("Unexpected end of WAV."); }
            return System.Text.Encoding.ASCII.GetString(tag);
        }
    }
}
