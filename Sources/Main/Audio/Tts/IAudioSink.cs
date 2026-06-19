// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Destination for synthesized mono PCM (samples in [−1, 1] at <see cref="SampleRate"/>). A
    /// <see cref="ITextToSpeechEngine"/> pushes audio in chunks as it decodes (so a backend can stream), then
    /// calls <see cref="Complete"/> once. Implementations: write to a WAV stream, a speaker, a network socket.
    /// </summary>
    public interface IAudioSink
    {
        /// <summary>Output sample rate in Hz.</summary>
        int SampleRate
        {
            get;
        }

        /// <summary>Appends a chunk of mono samples. May be called many times during one synthesis.</summary>
        void Write(ReadOnlySpan<float> samples);

        /// <summary>Signals end of synthesis — flush / finalize (e.g. write the WAV header). Idempotent.</summary>
        void Complete();
    }
}
