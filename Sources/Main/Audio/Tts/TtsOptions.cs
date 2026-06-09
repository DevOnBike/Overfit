// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>Per-call synthesis settings. Defaults target a 24 kHz neural-codec voice at natural pace.</summary>
    public sealed class TtsOptions
    {
        /// <summary>Output sample rate in Hz (the engine resamples / fixes its native rate if it must).</summary>
        public int SampleRate { get; init; } = 24000;

        /// <summary>Speaking-rate multiplier (1.0 = the model's natural pace).</summary>
        public float Speed { get; init; } = 1.0f;

        /// <summary>Sampling temperature for the acoustic model — higher is more varied, lower is flatter/steadier.</summary>
        public float Temperature { get; init; } = 0.7f;

        /// <summary>When true, the engine pushes audio to the sink in chunks as it decodes (lower latency).</summary>
        public bool Stream { get; init; } = true;

        /// <summary>The default settings.</summary>
        public static TtsOptions Default { get; } = new();
    }
}
