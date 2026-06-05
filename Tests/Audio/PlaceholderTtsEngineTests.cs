// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>The model-free placeholder engine drives the contract: it emits structured audio for text (so the
    /// CLI / pipeline is exercised now) and nothing for empty input. Samples stay in range.</summary>
    public sealed class PlaceholderTtsEngineTests
    {
        [Fact]
        public void Synthesize_ProducesAudio_AndStaysInRange()
        {
            var engine = new PlaceholderTtsEngine(24000);
            var sink = new CountingSink(engine.SampleRate);

            engine.Synthesize("hello world foo", VoiceProfile.Preset("v", "en"), sink, TtsOptions.Default);

            Assert.True(sink.Count > 0);
            Assert.True(sink.MaxAbs <= 1f, $"sample out of range: {sink.MaxAbs}");
        }

        [Fact]
        public void Synthesize_EmptyOrWhitespace_ProducesNoAudio()
        {
            var engine = new PlaceholderTtsEngine(24000);
            var sink = new CountingSink(engine.SampleRate);

            engine.Synthesize("   \t  ", VoiceProfile.Preset("v", "en"), sink, TtsOptions.Default);

            Assert.Equal(0, sink.Count);
        }

        private sealed class CountingSink(int sampleRate) : IAudioSink
        {
            public int SampleRate { get; } = sampleRate;
            public int Count { get; private set; }
            public float MaxAbs { get; private set; }

            public void Write(ReadOnlySpan<float> samples)
            {
                Count += samples.Length;
                for (var i = 0; i < samples.Length; i++)
                {
                    var a = MathF.Abs(samples[i]);
                    if (a > MaxAbs)
                    {
                        MaxAbs = a;
                    }
                }
            }

            public void Complete()
            {
            }
        }
    }
}
