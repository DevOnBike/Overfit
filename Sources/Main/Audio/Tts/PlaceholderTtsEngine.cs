// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// A model-free stand-in <see cref="ITextToSpeechEngine"/> until a real neural backend (the SNAC decoder +
    /// Orpheus glue) lands. It emits one short, enveloped tone per word — so the whole pipeline (CLI →
    /// synthesis → streamed PCM → watermarked WAV) is exercised end-to-end and the output is audibly "there".
    /// It is NOT speech; it exists to drive and validate the plumbing, not to sound like anything.
    /// </summary>
    public sealed class PlaceholderTtsEngine : ITextToSpeechEngine
    {
        private readonly float[] _scratch;

        public PlaceholderTtsEngine(int sampleRate = 24000)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
            SampleRate = sampleRate;
            _scratch = new float[sampleRate];   // up to ~1 s per tone
        }

        public int SampleRate { get; }

        public void Synthesize(ReadOnlySpan<char> text, VoiceProfile voice, IAudioSink output, TtsOptions options)
        {
            ArgumentNullException.ThrowIfNull(output);
            ArgumentNullException.ThrowIfNull(options);

            var speed = options.Speed <= 0f ? 1f : options.Speed;
            var amplitude = 0.25f;
            var gapSamples = (int)(0.06f * SampleRate / speed);

            var i = 0;
            while (i < text.Length)
            {
                while (i < text.Length && char.IsWhiteSpace(text[i]))
                {
                    i++;
                }
                if (i >= text.Length)
                {
                    break;
                }

                var start = i;
                var hash = 17;
                while (i < text.Length && !char.IsWhiteSpace(text[i]))
                {
                    hash = (hash * 31) + text[i];
                    i++;
                }

                var wordLen = i - start;
                EmitTone(output, wordLen, hash, amplitude, speed);
                EmitSilence(output, gapSamples);
            }
        }

        // One word → a sine tone whose length scales with the word and whose pitch is derived from the word,
        // shaped by a raised-cosine envelope so it doesn't click.
        private void EmitTone(IAudioSink output, int wordLen, int hash, float amplitude, float speed)
        {
            var durationSeconds = Math.Clamp(0.08f + wordLen * 0.045f, 0.08f, 0.9f) / speed;
            var count = Math.Min(_scratch.Length, (int)(durationSeconds * SampleRate));
            if (count <= 0)
            {
                return;
            }

            var frequency = 160f + ((hash & 0x7fffffff) % 6) * 40f;   // ~160–360 Hz, speech-ish
            var step = 2f * MathF.PI * frequency / SampleRate;
            var fade = Math.Max(1, count / 8);

            for (var n = 0; n < count; n++)
            {
                var envelope = 1f;
                if (n < fade)
                {
                    envelope = 0.5f * (1f - MathF.Cos(MathF.PI * n / fade));
                }
                else if (n >= count - fade)
                {
                    envelope = 0.5f * (1f - MathF.Cos(MathF.PI * (count - 1 - n) / fade));
                }
                _scratch[n] = amplitude * envelope * MathF.Sin(step * n);
            }

            output.Write(_scratch.AsSpan(0, count));
        }

        private void EmitSilence(IAudioSink output, int samples)
        {
            if (samples <= 0)
            {
                return;
            }
            var count = Math.Min(_scratch.Length, samples);
            _scratch.AsSpan(0, count).Clear();
            output.Write(_scratch.AsSpan(0, count));
        }
    }
}
