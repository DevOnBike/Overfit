// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>
    /// A parsed 32-bit MPEG audio frame header (Layer III only — this decoder targets MP3). Exposes the
    /// derived bitrate / sample-rate / frame-size / side-info-length needed to walk the bitstream.
    /// </summary>
    internal readonly struct Mp3FrameHeader
    {
        // Layer III bitrate (kbps) by [isMpeg1 ? 0 : 1][bitrate_index]. Index 0 = free, 15 = invalid.
        private static readonly int[][] BitrateKbps =
        {
            new[] { 0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, -1 }, // MPEG-1
            new[] { 0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, -1 },      // MPEG-2 / 2.5
        };

        // Sample rate (Hz) by [version][sr_index].
        private static readonly int[][] SampleRateHz =
        {
            new[] { 11025, 12000, 8000, 0 },  // MPEG 2.5
            new[] { 0, 0, 0, 0 },             // reserved
            new[] { 22050, 24000, 16000, 0 }, // MPEG 2
            new[] { 44100, 48000, 32000, 0 }, // MPEG 1
        };

        public MpegVersion Version
        {
            get;
        }
        public bool CrcProtected
        {
            get;
        }
        public int BitrateIndex
        {
            get;
        }
        public int SampleRateIndex
        {
            get;
        }
        public bool Padding
        {
            get;
        }
        public ChannelMode Mode
        {
            get;
        }
        public int ModeExtension
        {
            get;
        }

        private Mp3FrameHeader(MpegVersion version, bool crc, int bitrateIndex, int srIndex, bool padding, ChannelMode mode, int modeExt)
        {
            Version = version;
            CrcProtected = crc;
            BitrateIndex = bitrateIndex;
            SampleRateIndex = srIndex;
            Padding = padding;
            Mode = mode;
            ModeExtension = modeExt;
        }

        public bool IsMpeg1 => Version == MpegVersion.Mpeg1;
        public int Channels => Mode == ChannelMode.Mono ? 1 : 2;
        public int BitrateBps => BitrateKbps[IsMpeg1 ? 0 : 1][BitrateIndex] * 1000;
        public int SampleRate => SampleRateHz[(int)Version][SampleRateIndex];

        /// <summary>Samples per frame: 1152 (MPEG-1) or 576 (MPEG-2/2.5) for Layer III.</summary>
        public int SamplesPerFrame => IsMpeg1 ? 1152 : 576;

        /// <summary>Side-information length in bytes (after the 4-byte header + optional 2-byte CRC).</summary>
        public int SideInfoBytes => IsMpeg1 ? (Channels == 1 ? 17 : 32) : (Channels == 1 ? 9 : 17);

        /// <summary>Whole frame length in bytes (header included), per the ISO frame-size formula.</summary>
        public int FrameLengthBytes
        {
            get
            {
                var slotMul = IsMpeg1 ? 144 : 72; // 1152/8 vs 576/8
                return slotMul * BitrateBps / SampleRate + (Padding ? 1 : 0);
            }
        }

        /// <summary>
        /// Tries to parse a frame header from the 4 bytes at <paramref name="data"/>[<paramref name="offset"/>].
        /// Validates sync word, Layer III, and non-reserved/non-free fields.
        /// </summary>
        public static bool TryParse(byte[] data, int offset, out Mp3FrameHeader header)
        {
            header = default;
            if (offset + 4 > data.Length)
            {
                return false;
            }

            uint b0 = data[offset], b1 = data[offset + 1], b2 = data[offset + 2], b3 = data[offset + 3];

            // Sync: 11 set bits.
            if (b0 != 0xFF || (b1 & 0xE0) != 0xE0)
            {
                return false;
            }

            var version = (MpegVersion)((b1 >> 3) & 0x3);
            if (version == MpegVersion.Reserved)
            {
                return false;
            }

            var layer = (b1 >> 1) & 0x3; // 01 = Layer III
            if (layer != 0x1)
            {
                return false;
            }

            var crc = (b1 & 0x1) == 0; // protection bit: 0 means CRC present
            var bitrateIndex = (int)((b2 >> 4) & 0xF);
            var srIndex = (int)((b2 >> 2) & 0x3);
            if (bitrateIndex == 0 || bitrateIndex == 15 || srIndex == 3)
            {
                return false; // free-format / invalid not supported
            }

            var padding = ((b2 >> 1) & 0x1) != 0;
            var mode = (ChannelMode)((b3 >> 6) & 0x3);
            var modeExt = (int)((b3 >> 4) & 0x3);

            header = new Mp3FrameHeader(version, crc, bitrateIndex, srIndex, padding, mode, modeExt);
            return header.SampleRate > 0 && header.BitrateBps > 0;
        }
    }
}
