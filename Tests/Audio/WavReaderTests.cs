// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.IO;
using System.Text;
using DevOnBike.Overfit.Audio;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>
    /// Round-trips the dependency-free <see cref="WavReader"/>: a hand-built 16-bit PCM mono WAV reads
    /// back to the original samples (within quantization), and a stereo WAV down-mixes to mono.
    /// </summary>
    public sealed class WavReaderTests
    {
        [Fact]
        public void ReadMono_Pcm16_RoundTrips()
        {
            var samples = new float[1000];
            for (var i = 0; i < samples.Length; i++) { samples[i] = MathF.Sin(2f * MathF.PI * 5f * i / samples.Length) * 0.8f; }

            var wav = BuildPcm16Wav(samples, sampleRate: 16000, channels: 1);
            using var ms = new MemoryStream(wav);
            var read = WavReader.ReadMono(ms, out var sr);

            Assert.Equal(16000, sr);
            Assert.Equal(samples.Length, read.Length);
            for (var i = 0; i < samples.Length; i++)
            {
                Assert.True(Math.Abs(samples[i] - read[i]) < 1.0f / 32768 + 1e-4f, $"sample {i} off");
            }
        }

        [Fact]
        public void ReadMono_Stereo_DownmixesToMono()
        {
            // Two channels: L = 1.0, R = 0.0 → mono = 0.5.
            var interleaved = new float[4]; // 2 frames × 2 ch
            interleaved[0] = 1f; interleaved[1] = 0f;
            interleaved[2] = 1f; interleaved[3] = 0f;
            var wav = BuildPcm16Wav(interleaved, sampleRate: 16000, channels: 2);

            using var ms = new MemoryStream(wav);
            var mono = WavReader.ReadMono(ms, out _);

            Assert.Equal(2, mono.Length);
            Assert.True(Math.Abs(mono[0] - 0.5f) < 1e-3f);
            Assert.True(Math.Abs(mono[1] - 0.5f) < 1e-3f);
        }

        private static byte[] BuildPcm16Wav(float[] interleaved, int sampleRate, int channels)
        {
            var dataBytes = interleaved.Length * 2;
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            bw.Write(Encoding.ASCII.GetBytes("RIFF"));
            bw.Write(36 + dataBytes);
            bw.Write(Encoding.ASCII.GetBytes("WAVE"));
            bw.Write(Encoding.ASCII.GetBytes("fmt "));
            bw.Write(16);                                   // fmt chunk size
            bw.Write((short)1);                             // PCM
            bw.Write((short)channels);
            bw.Write(sampleRate);
            bw.Write(sampleRate * channels * 2);            // byte rate
            bw.Write((short)(channels * 2));                // block align
            bw.Write((short)16);                            // bits/sample
            bw.Write(Encoding.ASCII.GetBytes("data"));
            bw.Write(dataBytes);
            Span<byte> s = stackalloc byte[2];
            foreach (var v in interleaved)
            {
                var q = (short)Math.Clamp((int)MathF.Round(v * 32768f), short.MinValue, short.MaxValue);
                BinaryPrimitives.WriteInt16LittleEndian(s, q);
                bw.Write(s);
            }
            bw.Flush();
            return ms.ToArray();
        }
    }
}
