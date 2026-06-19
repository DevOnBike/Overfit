// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The SNAC neural audio codec — public entry point for turning codec tokens back into a waveform. This is the
    /// decoder half (codes → PCM): the piece Orpheus-style TTS needs to vocode the audio tokens an LLM emits, in
    /// pure managed .NET on the CPU. Load once from the converted weights, then <see cref="Decode"/> per utterance.
    /// </summary>
    public sealed class Snac
    {
        private readonly SnacDecoder _decoder;
        private readonly SnacEncoder _encoder;

        private Snac(SnacConfig config, SnacWeights weights)
        {
            Config = config;
            _decoder = new SnacDecoder(config, weights);
            _encoder = new SnacEncoder(config, weights);
        }

        public SnacConfig Config
        {
            get;
        }

        public int SampleRate => Config.SampleRate;

        /// <summary>
        /// Loads the SNAC 24 kHz decoder from a directory containing <c>snac_24khz.safetensors</c> (produced by
        /// <c>Scripts/convert_snac.py</c>).
        /// </summary>
        public static Snac Load(string directory)
        {
            var path = Path.Combine(directory, "snac_24khz.safetensors");
            if (!File.Exists(path))
            {
                throw new OverfitFormatException(
                    $"SNAC weights not found at '{path}'. Run Scripts/convert_snac.py --out \"{directory}\".");
            }
            return new Snac(SnacConfig.Snac24Khz, SnacWeights.Load(path));
        }

        /// <summary>
        /// Decodes residual-VQ codes (one int array per level, coarse→fine) to mono PCM in <c>[-1, 1]</c> at
        /// <see cref="SampleRate"/>. <paramref name="addNoise"/> enables SNAC's stochastic NoiseBlock; leave it off
        /// for a deterministic, reproducible decode.
        /// </summary>
        public float[] Decode(int[][] codes, bool addNoise = false) => _decoder.Decode(codes, addNoise);

        /// <summary>
        /// Encodes mono 24 kHz PCM in <c>[-1, 1]</c> into residual-VQ codes (one int array per level, coarse→fine)
        /// — the inverse of <see cref="Decode"/>. Useful for round-tripping audio and for preparing
        /// audio-token training data (e.g. voice-clone fine-tuning) entirely in managed .NET.
        /// </summary>
        public int[][] Encode(ReadOnlySpan<float> audio) => _encoder.Encode(audio);
    }
}
