// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Minimal, dependency-free WAV writer — the inverse of <see cref="WavReader"/>. Writes mono float samples
    /// (in [−1, 1]) as a standard RIFF/WAVE file in 16-bit PCM (default) or 32-bit IEEE float. Optionally embeds a
    /// RIFF <c>LIST/INFO</c> chunk (software + comment), used to carry a synthetic-speech provenance marker; the
    /// reader skips it, so the audio round-trips unchanged. The TTS output path; pairs with <see cref="WavReader"/>.
    /// </summary>
    public static class WavWriter
    {
        /// <summary>Writes mono <paramref name="samples"/> to <paramref name="path"/>.</summary>
        public static void WriteMono(string path, ReadOnlySpan<float> samples, int sampleRate,
            WavSampleFormat format = WavSampleFormat.Pcm16, string? infoComment = null)
        {
            using var fs = File.Create(path);
            WriteMono(fs, samples, sampleRate, format, infoComment);
        }

        /// <summary>Writes mono <paramref name="samples"/> as a RIFF/WAVE stream.</summary>
        public static void WriteMono(Stream stream, ReadOnlySpan<float> samples, int sampleRate,
            WavSampleFormat format = WavSampleFormat.Pcm16, string? infoComment = null)
        {
            ArgumentNullException.ThrowIfNull(stream);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);

            var bytesPerSample = format == WavSampleFormat.Float32 ? 4 : 2;
            var audioFormat = format == WavSampleFormat.Float32 ? (short)3 : (short)1;
            var bitsPerSample = (short)(bytesPerSample * 8);
            const short channels = 1;
            var blockAlign = (short)(channels * bytesPerSample);
            var byteRate = sampleRate * blockAlign;
            var dataBytes = samples.Length * bytesPerSample;     // mono → always even

            var infoChunk = infoComment is null ? [] : BuildInfoList(infoComment);

            // RIFF size = "WAVE"(4) + fmt(8+16) + LIST(infoChunk) + data(8 + dataBytes).
            var riffSize = 4 + (8 + 16) + infoChunk.Length + (8 + dataBytes);

            using var bw = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);

            WriteTag(bw, "RIFF");
            bw.Write(riffSize);
            WriteTag(bw, "WAVE");

            WriteTag(bw, "fmt ");
            bw.Write(16);                 // fmt chunk size
            bw.Write(audioFormat);
            bw.Write(channels);
            bw.Write(sampleRate);
            bw.Write(byteRate);
            bw.Write(blockAlign);
            bw.Write(bitsPerSample);

            if (infoChunk.Length > 0)
            {
                bw.Write(infoChunk);
            }

            WriteTag(bw, "data");
            bw.Write(dataBytes);
            if (format == WavSampleFormat.Float32)
            {
                for (var i = 0; i < samples.Length; i++)
                {
                    bw.Write(samples[i]);
                }
            }
            else
            {
                for (var i = 0; i < samples.Length; i++)
                {
                    var clamped = samples[i] < -1f ? -1f : (samples[i] > 1f ? 1f : samples[i]);
                    bw.Write((short)MathF.Round(clamped * 32767f));
                }
            }
        }

        // A RIFF LIST/INFO chunk: software ("ISFT" = Overfit) + a free-text comment ("ICMT"). Word-aligned.
        private static byte[] BuildInfoList(string comment)
        {
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true))
            {
                WriteTag(bw, "LIST");
                var sizePos = ms.Position;
                bw.Write(0);                  // LIST size — patched below
                WriteTag(bw, "INFO");
                WriteInfoField(bw, "ISFT", "Overfit");
                WriteInfoField(bw, "ICMT", comment);

                var end = ms.Position;
                var listSize = (int)(end - sizePos - 4);
                ms.Position = sizePos;
                bw.Write(listSize);
                ms.Position = end;
            }
            return ms.ToArray();
        }

        private static void WriteInfoField(BinaryWriter bw, string id, string value)
        {
            WriteTag(bw, id);
            var bytes = Encoding.UTF8.GetBytes(value);
            var size = bytes.Length + 1;      // include the null terminator
            bw.Write(size);
            bw.Write(bytes);
            bw.Write((byte)0);                // null terminator
            if ((size & 1) == 1)
            {
                bw.Write((byte)0);            // pad to even
            }
        }

        private static void WriteTag(BinaryWriter bw, string tag)
        {
            // Tags are exactly 4 ASCII chars.
            bw.Write((byte)tag[0]);
            bw.Write((byte)tag[1]);
            bw.Write((byte)tag[2]);
            bw.Write((byte)tag[3]);
        }
    }
}
