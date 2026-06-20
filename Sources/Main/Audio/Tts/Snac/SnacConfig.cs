// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Tts.Snac
{
    /// <summary>
    /// The fixed architecture of the SNAC 24 kHz codec (the variant Overfit's decoder supports — the one Orpheus
    /// TTS emits). Constants, not a free-form config, because the decoder graph is wired to exactly this shape;
    /// other SNAC sizes (44 kHz) would need their own profile.
    /// </summary>
    public sealed class SnacConfig
    {
        private SnacConfig()
        {
        }

        public int SampleRate
        {
            get; private init;
        }

        /// <summary>The quantizer / decoder-input channel count (encoder_dim·2^len(encoder_rates) = 48·16).</summary>
        public int LatentDim
        {
            get; private init;
        }

        /// <summary>The encoder's first conv width (doubles per downsampling block).</summary>
        public int EncoderDim
        {
            get; private init;
        }

        /// <summary>Per-block downsampling factors of the encoder (mirror of <see cref="DecoderRates"/>).</summary>
        public int[] EncoderRates { get; private init; } = [];

        /// <summary>The decoder's widest channel count, after the input pointwise conv.</summary>
        public int DecoderDim
        {
            get; private init;
        }

        /// <summary>Per-block upsampling factors; their product is the decode hop (512 samples / frame).</summary>
        public int[] DecoderRates { get; private init; } = [];

        public int CodebookSize
        {
            get; private init;
        }
        public int CodebookDim
        {
            get; private init;
        }

        /// <summary>One per residual-VQ level: the temporal stride each level was quantized at (so it is
        /// repeat-interleaved back up before the cross-level sum).</summary>
        public int[] VqStrides { get; private init; } = [];

        public int CodebookCount => VqStrides.Length;

        /// <summary>Total decode upsampling factor (∏ DecoderRates) — output samples per latent frame.</summary>
        public int HopLength
        {
            get
            {
                var h = 1;
                foreach (var r in DecoderRates)
                {
                    h *= r;
                }
                return h;
            }
        }

        /// <summary>The published <c>hubertsiuzdak/snac_24khz</c> profile.</summary>
        public static SnacConfig Snac24Khz
        {
            get;
        } = new()
        {
            SampleRate = 24_000,
            LatentDim = 768,
            EncoderDim = 48,
            EncoderRates = [2, 4, 8, 8],
            DecoderDim = 1024,
            DecoderRates = [8, 8, 4, 2],
            CodebookSize = 4096,
            CodebookDim = 8,
            VqStrides = [4, 2, 1],
        };
    }
}
