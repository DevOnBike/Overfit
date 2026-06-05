// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Provenance marker for synthesized speech — written into the WAV's <c>LIST/INFO</c> comment so any media
    /// tool (and any downstream consumer) can see the audio is machine-generated and by whom. Disclosing
    /// synthetic speech is a legal obligation in a growing number of jurisdictions (e.g. the EU AI Act's
    /// transparency rules); emitting this marker is part of doing voice synthesis responsibly, not optional polish.
    /// </summary>
    public sealed class SyntheticSpeechMetadata
    {
        public SyntheticSpeechMetadata(string? voiceProfileId, string createdUtc)
        {
            ArgumentNullException.ThrowIfNull(createdUtc);
            VoiceProfileId = voiceProfileId;
            CreatedUtc = createdUtc;
        }

        /// <summary>The producing engine.</summary>
        public string GeneratedBy => "Overfit";

        /// <summary>Id of the voice used (null for a default/preset voice).</summary>
        public string? VoiceProfileId { get; }

        /// <summary>ISO-8601 UTC timestamp the caller stamped at synthesis time.</summary>
        public string CreatedUtc { get; }

        /// <summary>Always true — this audio is synthetic.</summary>
        public bool SyntheticSpeech => true;

        /// <summary>Renders the marker as the WAV <c>ICMT</c> comment string.</summary>
        public string ToInfoComment()
            => $"generatedBy=Overfit; synthetic=true; voice={VoiceProfileId ?? "-"}; createdUtc={CreatedUtc}";

        /// <summary>Convenience: a marker stamped with the current UTC time.</summary>
        public static SyntheticSpeechMetadata ForNow(string? voiceProfileId)
            => new(voiceProfileId, DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture));
    }
}
