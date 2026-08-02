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
        public string? VoiceProfileId
        {
            get;
        }

        /// <summary>ISO-8601 UTC timestamp the caller stamped at synthesis time.</summary>
        public string CreatedUtc
        {
            get;
        }

        /// <summary>Always true — this audio is synthetic.</summary>
        public bool SyntheticSpeech => true;

        /// <summary>Renders the marker as the WAV <c>ICMT</c> comment string.</summary>
        /// <summary>
        /// Whether this instance actually marks the output. False only for
        /// <see cref="Unmarked"/>.
        /// </summary>
        public bool Marks => !ReferenceEquals(this, Unmarked);

        /// <summary>
        /// An explicit decision to write audio without the synthetic marker.
        ///
        /// <para>It exists so that unmarked output requires naming this property at a call site, where a
        /// reviewer can see it, rather than being what happens when an optional argument is left out. Sinks
        /// treat a missing marker as "mark it"; they treat this as "the caller decided".</para>
        /// </summary>
        public static SyntheticSpeechMetadata Unmarked { get; } = new(null, string.Empty);

        public string ToInfoComment()
            => Marks
                ? $"generatedBy=Overfit; synthetic=true; voice={VoiceProfileId ?? "-"}; createdUtc={CreatedUtc}"
                : string.Empty;

        /// <summary>Convenience: a marker stamped with the current UTC time.</summary>
        public static SyntheticSpeechMetadata ForNow(string? voiceProfileId)
            => new(voiceProfileId, DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture));
    }
}
