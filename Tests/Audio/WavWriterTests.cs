// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>
    /// The TTS output path is testable now, with no model: <see cref="WavWriter"/> must round-trip through
    /// <see cref="WavReader"/>, and the synthetic-speech provenance marker must be embedded without breaking the
    /// audio.
    /// </summary>
    public sealed class WavWriterTests
    {
        [Fact]
        public void WriteThenRead_Pcm16_RoundTripsWithinQuantization()
        {
            float[] samples = [0f, 0.5f, -0.5f, 0.99f, -0.99f, 0.123f];

            using var ms = new MemoryStream();
            WavWriter.WriteMono(ms, samples, 24000, WavSampleFormat.Pcm16);
            ms.Position = 0;

            var read = WavReader.ReadMono(ms, out var rate);

            Assert.Equal(24000, rate);
            Assert.Equal(samples.Length, read.Length);
            for (var i = 0; i < samples.Length; i++)
            {
                Assert.True(MathF.Abs(samples[i] - read[i]) < 2e-4f, $"i={i}: {samples[i]} vs {read[i]}");
            }
        }

        [Fact]
        public void WriteThenRead_Float32_RoundTripsExactly()
        {
            float[] samples = [0f, 0.5f, -0.123456f, 0.999f, -1f, 1f];

            using var ms = new MemoryStream();
            WavWriter.WriteMono(ms, samples, 16000, WavSampleFormat.Float32);
            ms.Position = 0;

            var read = WavReader.ReadMono(ms, out var rate);

            Assert.Equal(16000, rate);
            Assert.Equal(samples, read);
        }

        [Fact]
        public void InfoComment_IsEmbedded_AndAudioStillReadsBack()
        {
            float[] samples = [0.1f, 0.2f, 0.3f];
            var meta = new SyntheticSpeechMetadata("maciej", "2026-06-05T00:00:00.0000000Z");

            using var ms = new MemoryStream();
            WavWriter.WriteMono(ms, samples, 24000, WavSampleFormat.Pcm16, meta.ToInfoComment());

            var text = Encoding.ASCII.GetString(ms.ToArray());
            Assert.Contains("INFO", text);
            Assert.Contains("Overfit", text);
            Assert.Contains("synthetic=true", text);
            Assert.Contains("voice=maciej", text);

            // The reader skips the LIST/INFO chunk, so the audio still round-trips.
            ms.Position = 0;
            var read = WavReader.ReadMono(ms, out _);
            Assert.Equal(3, read.Length);
        }
    }
}
