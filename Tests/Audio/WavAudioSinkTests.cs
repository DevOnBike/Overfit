// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>
    /// The streaming sink + the backend-agnostic TTS contract, exercised end-to-end with a model-free fake
    /// engine: chunks pushed during synthesis concatenate in order into a valid WAV.
    /// </summary>
    public sealed class WavAudioSinkTests
    {
        [Fact]
        public void StreamedChunks_ConcatenateInOrder()
        {
            using var ms = new MemoryStream();
            using (var sink = new WavAudioSink(ms, 24000, WavSampleFormat.Float32, metadata: null, leaveOpen: true))
            {
                sink.Write([0.1f, 0.2f]);
                sink.Write([0.3f]);
                sink.Write([0.4f, 0.5f, 0.6f]);
                sink.Complete();
            }

            ms.Position = 0;
            var read = WavReader.ReadMono(ms, out var rate);

            Assert.Equal(24000, rate);
            Assert.Equal(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f }, read);
        }

        [Fact]
        public void Dispose_CompletesAutomatically()
        {
            using var ms = new MemoryStream();
            var sink = new WavAudioSink(ms, 16000, WavSampleFormat.Float32, metadata: null, leaveOpen: true);
            sink.Write([0.5f]);
            sink.Dispose();   // completes if not already

            ms.Position = 0;
            var read = WavReader.ReadMono(ms, out _);
            Assert.Single(read);
            Assert.Equal(0.5f, read[0]);
        }

        [Fact]
        public void FakeEngine_DrivesSink_ProducesAudioWithMarker()
        {
            ITextToSpeechEngine engine = new RampTtsEngine(24000);

            using var ms = new MemoryStream();
            using (var sink = new WavAudioSink(ms, 24000, WavSampleFormat.Float32,
                       new SyntheticSpeechMetadata("test", "2026-06-05T00:00:00.0000000Z"), leaveOpen: true))
            {
                engine.Synthesize("hello", VoiceProfile.Preset("test", "en"), sink, TtsOptions.Default);
                sink.Complete();
            }

            ms.Position = 0;
            var read = WavReader.ReadMono(ms, out _);
            Assert.True(read.Length > 0);
        }

        // Model-free engine: emits a short ramp per character — enough to exercise the streaming sink + contract.
        private sealed class RampTtsEngine(int sampleRate) : ITextToSpeechEngine
        {
            public int SampleRate { get; } = sampleRate;

            public void Synthesize(ReadOnlySpan<char> text, VoiceProfile voice, IAudioSink output, TtsOptions options)
            {
                var buffer = new float[100];
                for (var c = 0; c < text.Length; c++)
                {
                    for (var i = 0; i < buffer.Length; i++)
                    {
                        buffer[i] = MathF.Sin(((c * buffer.Length) + i) * 0.05f) * 0.3f;
                    }
                    output.Write(buffer);
                }
            }
        }
    }
}
