// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Backend-agnostic text-to-speech contract. A concrete engine (an LLM + neural-codec stack such as
    /// Orpheus + SNAC, or another model) turns <paramref name="text"/> into speech in the given
    /// <see cref="VoiceProfile"/> and pushes the mono PCM to <paramref name="output"/>. Keeping this interface
    /// model-independent lets the runtime, enrollment, demos and the voice loop be built and tested before any
    /// specific model is ported.
    /// </summary>
    public interface ITextToSpeechEngine
    {
        /// <summary>The engine's native output sample rate in Hz.</summary>
        int SampleRate
        {
            get;
        }

        /// <summary>Synthesizes <paramref name="text"/> in <paramref name="voice"/> into <paramref name="output"/>.</summary>
        void Synthesize(ReadOnlySpan<char> text, VoiceProfile voice, IAudioSink output, TtsOptions options);
    }
}
