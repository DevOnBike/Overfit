// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// The voice a <see cref="ITextToSpeechEngine"/> speaks in. A <b>preset</b> voice is just an id + language; a
    /// <b>cloned</b> voice additionally carries a <see cref="SpeakerEmbedding"/> (a conditioning vector derived
    /// from a short reference clip) and/or the <see cref="ReferenceAudioPath"/> the embedding came from. Voice
    /// cloning is only legitimate for a voice you own or have explicit consent to use.
    /// </summary>
    public sealed class VoiceProfile
    {
        public VoiceProfile(
            string id,
            string language = "en",
            float[]? speakerEmbedding = null,
            string? referenceAudioPath = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(id);
            ArgumentException.ThrowIfNullOrWhiteSpace(language);
            Id = id;
            Language = language;
            SpeakerEmbedding = speakerEmbedding;
            ReferenceAudioPath = referenceAudioPath;
        }

        /// <summary>Stable identifier for the voice (also stamped into the synthetic-speech marker).</summary>
        public string Id { get; }

        /// <summary>BCP-47-ish language tag the text is in (e.g. <c>"pl"</c>, <c>"en"</c>).</summary>
        public string Language { get; }

        /// <summary>Speaker-conditioning vector for a cloned voice; null for a preset voice.</summary>
        public float[]? SpeakerEmbedding { get; }

        /// <summary>Path to the reference clip the embedding was enrolled from, if any.</summary>
        public string? ReferenceAudioPath { get; }

        /// <summary>True when this profile carries cloning conditioning (an enrolled speaker embedding).</summary>
        public bool IsCloned => SpeakerEmbedding is { Length: > 0 };

        /// <summary>A preset (non-cloned) voice — an id + language only.</summary>
        public static VoiceProfile Preset(string id, string language = "en") => new(id, language);
    }
}
