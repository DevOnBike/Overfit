// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>
    /// Constant tables for MPEG Layer III decoding, taken from ISO/IEC 11172-3 (MPEG-1) and 13818-3 (MPEG-2 LSF):
    /// scalefactor-band boundaries, MPEG-2 LSF scalefactor partitioning, IMDCT windows, the subband-synthesis
    /// cosine matrix, and the D[512] synthesis window. Pure data — computed where a closed form exists
    /// (windows, cosine matrices), tabulated where the standard tabulates (band boundaries, D[]).
    /// </summary>
    // OVERFIT001 (per-call array alloc): every allocation in this file is a `static readonly` decode-table
    // builder run ONCE at type initialization — pure load-time constant data, not a per-call hot path.
#pragma warning disable OVERFIT001
    internal static class Mp3Tables
    {
        /// <summary>
        /// Maps (version, sample_rate_index) → a 0..8 scalefactor-band table index:
        /// 0,1,2 = MPEG-1 44100/48000/32000; 3,4,5 = MPEG-2 22050/24000/16000; 6,7,8 = MPEG-2.5 11025/12000/8000.
        /// </summary>
        public static int SfBandTableIndex(MpegVersion version, int srIndex) => version switch
        {
            MpegVersion.Mpeg1 => srIndex,
            MpegVersion.Mpeg2 => 3 + srIndex,
            _ => 6 + srIndex, // Mpeg25
        };

        /// <summary>Long-block scalefactor-band boundaries (23 entries: 22 bands + the 576 terminator), per table index.</summary>
        public static readonly int[][] SfBandLong =
        {
            new[] { 0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576 }, // 44100
            new[] { 0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384, 576 }, // 48000
            new[] { 0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550, 576 }, // 32000
            new[] { 0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576 }, // 22050
            new[] { 0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 114, 136, 162, 194, 232, 278, 332, 394, 464, 540, 576 }, // 24000
            new[] { 0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576 }, // 16000
            new[] { 0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576 }, // 11025
            new[] { 0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576 }, // 12000
            new[] { 0, 12, 24, 36, 48, 60, 72, 88, 108, 132, 160, 192, 232, 280, 336, 400, 476, 566, 568, 570, 572, 574, 576 }, // 8000
        };

        /// <summary>Short-block scalefactor-band widths' boundaries (13 bands → 14 entries, range 0..192), per table index.</summary>
        public static readonly int[][] SfBandShort =
        {
            new[] { 0, 4, 8, 12, 16, 22, 30, 40, 52, 66, 84, 106, 136, 192 }, // 44100
            new[] { 0, 4, 8, 12, 16, 22, 28, 38, 50, 64, 80, 100, 126, 192 }, // 48000
            new[] { 0, 4, 8, 12, 16, 22, 30, 42, 58, 78, 104, 138, 180, 192 }, // 32000
            new[] { 0, 4, 8, 12, 18, 24, 32, 42, 56, 74, 100, 132, 174, 192 }, // 22050
            new[] { 0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 136, 180, 192 }, // 24000
            new[] { 0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 134, 174, 192 }, // 16000
            new[] { 0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 134, 174, 192 }, // 11025
            new[] { 0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 134, 174, 192 }, // 12000
            new[] { 0, 8, 16, 24, 36, 52, 72, 96, 124, 160, 162, 164, 166, 192 }, // 8000
        };

        // ── IMDCT windows (block types 0=normal, 1=start, 2=short, 3=stop) ──
        // Flat [4 × 36], indexed ImdctWin[blockType*36 + i]. Long windows are 36-point; the short window is
        // 12-point (<see cref="ImdctShortWin"/>), applied to 3 sub-blocks.
        public const int ImdctWinStride = 36;
        public static readonly float[] ImdctWin = BuildImdctWindows();
        public static readonly float[] ImdctShortWin = BuildShortWindow();

        private static float[] BuildImdctWindows()
        {
            var win = new float[4 * 36];

            // Type 0 — normal: sin(pi/36 * (i + 0.5))
            for (var i = 0; i < 36; i++)
            {
                win[0 * 36 + i] = MathF.Sin(MathF.PI / 36f * (i + 0.5f));
            }

            // Type 1 — start block
            for (var i = 0; i < 18; i++)
            {
                win[1 * 36 + i] = MathF.Sin(MathF.PI / 36f * (i + 0.5f));
            }
            for (var i = 18; i < 24; i++)
            {
                win[1 * 36 + i] = 1f;
            }
            for (var i = 24; i < 30; i++)
            {
                win[1 * 36 + i] = MathF.Sin(MathF.PI / 12f * (i - 18 + 0.5f));
            }
            for (var i = 30; i < 36; i++)
            {
                win[1 * 36 + i] = 0f;
            }

            // Type 3 — stop block
            for (var i = 0; i < 6; i++)
            {
                win[3 * 36 + i] = 0f;
            }
            for (var i = 6; i < 12; i++)
            {
                win[3 * 36 + i] = MathF.Sin(MathF.PI / 12f * (i - 6 + 0.5f));
            }
            for (var i = 12; i < 18; i++)
            {
                win[3 * 36 + i] = 1f;
            }
            for (var i = 18; i < 36; i++)
            {
                win[3 * 36 + i] = MathF.Sin(MathF.PI / 36f * (i + 0.5f));
            }

            // Type 2 — short: sin(pi/12*(i+0.5)) for the first 12 points, 0 after (as in the reference IMDCT).
            for (var i = 0; i < 12; i++)
            {
                win[2 * 36 + i] = MathF.Sin(MathF.PI / 12f * (i + 0.5f));
            }
            for (var i = 12; i < 36; i++)
            {
                win[2 * 36 + i] = 0f;
            }
            return win;
        }

        // ── IMDCT cosine tables: cos(pi/(2N)·(2p+1+N/2)·(2m+1)) for N=36 (long) and N=12 (short) ──
        public static readonly float[] CosN36 = BuildCos(36); // [18 (m) × 36 (p)]
        public static readonly float[] CosN12 = BuildCos(12); // [6  (m) × 12 (p)]

        private static float[] BuildCos(int n)
        {
            var half = n / 2;
            var t = new float[half * n];
            for (var m = 0; m < half; m++)
            {
                for (var p = 0; p < n; p++)
                {
                    t[m * n + p] = MathF.Cos(MathF.PI / (2 * n) * (2 * p + 1 + half) * (2 * m + 1));
                }
            }
            return t;
        }

        private static float[] BuildShortWindow()
        {
            var w = new float[12];
            for (var i = 0; i < 12; i++)
            {
                w[i] = MathF.Sin(MathF.PI / 12f * (i + 0.5f));
            }
            return w;
        }

        // ── Antialias butterfly coefficients (cs/ca for the 8 butterflies) ──
        private static readonly float[] AliasCi = { -0.6f, -0.535f, -0.33f, -0.185f, -0.095f, -0.041f, -0.0142f, -0.0037f };
        public static readonly float[] AliasCs = BuildAliasCs();
        public static readonly float[] AliasCa = BuildAliasCa();

        private static float[] BuildAliasCs()
        {
            var cs = new float[8];
            for (var i = 0; i < 8; i++)
            {
                cs[i] = 1f / MathF.Sqrt(1f + AliasCi[i] * AliasCi[i]);
            }
            return cs;
        }

        private static float[] BuildAliasCa()
        {
            var ca = new float[8];
            for (var i = 0; i < 8; i++)
            {
                ca[i] = AliasCi[i] / MathF.Sqrt(1f + AliasCi[i] * AliasCi[i]);
            }
            return ca;
        }

        // ── Subband-synthesis cosine matrix N[i][k] = cos((16+i)*(2k+1)*pi/64) ──
        public static readonly float[] SynthN = BuildSynthN();

        private static float[] BuildSynthN()
        {
            var n = new float[64 * 32];
            for (var i = 0; i < 64; i++)
            {
                for (var k = 0; k < 32; k++)
                {
                    n[i * 32 + k] = MathF.Cos((16 + i) * (2 * k + 1) * MathF.PI / 64f);
                }
            }
            return n;
        }

        /// <summary>
        /// The D[512] subband-synthesis window from ISO 11172-3 Table 3-B.3 (the canonical table lives in
        /// <see cref="Mp3SynthWindowData"/>).
        /// </summary>
        public static readonly float[] SynthWindow = Mp3SynthWindowData.D;
    }
#pragma warning restore OVERFIT001
}
