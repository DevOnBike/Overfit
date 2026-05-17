// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Pure byte→float decoders for GGML K-quants (Q4_K, Q6_K). Algorithms mirror
    /// llama.cpp ggml-quants.c exactly so produced floats match llama.cpp/Ollama
    /// bit-for-bit (modulo FP16→FP32 conversion).
    ///
    /// Functions are pure: no I/O, no allocations. Callers handle streaming.
    /// All blocks are QK_K=256 elements (the only super-block size used in real
    /// GGUF files).
    /// </summary>
    internal static class GgmlDequant
    {
        /// <summary>Elements per K-quant super-block.</summary>
        public const int SuperBlockElements = 256;

        /// <summary>Bytes per Q4_K super-block: 2(d) + 2(dmin) + 12(scales/mins) + 128(nibbles).</summary>
        public const int Q4_K_BlockBytes = 144;

        /// <summary>Bytes per Q6_K super-block: 128(ql) + 64(qh) + 16(scales) + 2(d).</summary>
        public const int Q6_K_BlockBytes = 210;

        /// <summary>
        /// Unpacks the 12-byte packed scales/mins of a Q4_K super-block into 8 scales
        /// and 8 mins (each 6 bits, range 0..63). Layout from llama.cpp ggml-quants.c
        /// <c>get_scale_min_k4</c>.
        ///
        /// For sub-block j ∈ [0..7]:
        ///   j &lt; 4 : scale = q[j]   &amp; 0x3F;             min = q[j+4]   &amp; 0x3F
        ///   j ≥ 4 : scale = (q[j+4] &amp; 0x0F) | ((q[j-4] &gt;&gt; 6) &lt;&lt; 4)
        ///           min   = (q[j+4] &gt;&gt; 4)   | ((q[j]   &gt;&gt; 6) &lt;&lt; 4)
        /// </summary>
        public static void UnpackQ4_KScalesMins(
            ReadOnlySpan<byte> packed12,
            Span<byte> scales8,
            Span<byte> mins8)
        {
            if (packed12.Length != 12)
            {
                throw new ArgumentException("packed12 must be exactly 12 bytes.", nameof(packed12));
            }
            if (scales8.Length != 8)
            {
                throw new ArgumentException("scales8 must be exactly 8 bytes.", nameof(scales8));
            }
            if (mins8.Length != 8)
            {
                throw new ArgumentException("mins8 must be exactly 8 bytes.", nameof(mins8));
            }

            for (var j = 0; j < 4; j++)
            {
                scales8[j] = (byte)(packed12[j] & 0x3F);
                mins8[j] = (byte)(packed12[j + 4] & 0x3F);
            }
            for (var j = 4; j < 8; j++)
            {
                scales8[j] = (byte)((packed12[j + 4] & 0x0F) | ((packed12[j - 4] >> 6) << 4));
                mins8[j] = (byte)((packed12[j + 4] >> 4) | ((packed12[j] >> 6) << 4));
            }
        }

        /// <summary>
        /// Decodes a single Q4_K super-block (144 bytes) into 256 floats.
        ///
        /// Block layout (from ggml-quants.c block_q4_K):
        ///   [0..1]    d        — FP16 super-scale for the 6-bit scales
        ///   [2..3]    dmin     — FP16 super-scale for the 6-bit mins
        ///   [4..15]   12 bytes — packed scales[8] + mins[8] (6 bits each)
        ///   [16..143] 128 bytes — 256 4-bit nibbles (low/high in each byte)
        ///
        /// Decode (mirrors dequantize_row_q4_K, processing 64 elements per sub-pair):
        ///   For sub-pair p ∈ [0..3]:
        ///     scale1 = d * scales[2p+0];   min1 = dmin * mins[2p+0]
        ///     scale2 = d * scales[2p+1];   min2 = dmin * mins[2p+1]
        ///     First 32 elements:  dst[i] = scale1 * (qs[32p + i] &amp; 0x0F) - min1
        ///     Next 32 elements:   dst[i] = scale2 * (qs[32p + i] &gt;&gt; 4)   - min2
        /// </summary>
        public static void DecodeQ4_KBlock(ReadOnlySpan<byte> block144, Span<float> dst256)
        {
            if (block144.Length != Q4_K_BlockBytes)
            {
                throw new ArgumentException(
                    $"Q4_K block must be exactly {Q4_K_BlockBytes} bytes.", nameof(block144));
            }
            if (dst256.Length != SuperBlockElements)
            {
                throw new ArgumentException(
                    $"dst must be exactly {SuperBlockElements} floats.", nameof(dst256));
            }

            var d = (float)BitConverter.UInt16BitsToHalf(
                BinaryPrimitives.ReadUInt16LittleEndian(block144[..2]));
            var dmin = (float)BitConverter.UInt16BitsToHalf(
                BinaryPrimitives.ReadUInt16LittleEndian(block144.Slice(2, 2)));

            Span<byte> scales = stackalloc byte[8];
            Span<byte> mins = stackalloc byte[8];
            UnpackQ4_KScalesMins(block144.Slice(4, 12), scales, mins);

            var qs = block144.Slice(16, 128);  // 256 nibbles in 128 bytes

            // Four sub-pairs × 64 elements each = 256.
            for (var p = 0; p < 4; p++)
            {
                var scale1 = d * scales[2 * p + 0];
                var min1 = dmin * mins[2 * p + 0];
                var scale2 = d * scales[2 * p + 1];
                var min2 = dmin * mins[2 * p + 1];

                var qsBase = 32 * p;
                var dstBase = 64 * p;

                // First 32: low nibbles
                for (var i = 0; i < 32; i++)
                {
                    dst256[dstBase + i] = scale1 * (qs[qsBase + i] & 0x0F) - min1;
                }
                // Next 32: high nibbles
                for (var i = 0; i < 32; i++)
                {
                    dst256[dstBase + 32 + i] = scale2 * (qs[qsBase + i] >> 4) - min2;
                }
            }
        }

        /// <summary>
        /// Decodes a single Q6_K super-block (210 bytes) into 256 floats.
        ///
        /// Block layout (from ggml-quants.c block_q6_K):
        ///   [0..127]   ql       — 256 lower 4-bit quants (low/high in each byte)
        ///   [128..191] qh       — 256 upper 2-bit quants (4 quants per byte)
        ///   [192..207] scales   — 16 × int8 (signed!), one per 16-element sub-block
        ///   [208..209] d        — FP16 super-scale
        ///
        /// 6-bit quant assembly (for element l in a 128-element half-block):
        ///   q = (ql[l] &amp; 0x0F) | (((qh[l] &gt;&gt; shift) &amp; 0x03) &lt;&lt; 4)   — range [0..63]
        ///   then biased: q − 32                                          — range [-32..31]
        ///   shift ∈ {0, 2, 4, 6} cycles through the 4 quants packed in qh[l]
        ///
        /// Output: dst[i] = d * scales[sub_block_of(i)] * (q − 32)
        /// </summary>
        public static void DecodeQ6_KBlock(ReadOnlySpan<byte> block210, Span<float> dst256)
        {
            if (block210.Length != Q6_K_BlockBytes)
            {
                throw new ArgumentException(
                    $"Q6_K block must be exactly {Q6_K_BlockBytes} bytes.", nameof(block210));
            }
            if (dst256.Length != SuperBlockElements)
            {
                throw new ArgumentException(
                    $"dst must be exactly {SuperBlockElements} floats.", nameof(dst256));
            }

            var ql = block210[..128];
            var qh = block210.Slice(128, 64);
            var sc = block210.Slice(192, 16);
            var d = (float)BitConverter.UInt16BitsToHalf(
                BinaryPrimitives.ReadUInt16LittleEndian(block210.Slice(208, 2)));

            // Two half-blocks of 128 elements; in each, write 4 strides of 32.
            // Half h ∈ {0,1}: uses ql[h*64..h*64+64], qh[h*32..h*32+32], sc[h*8..h*8+8].
            for (var h = 0; h < 2; h++)
            {
                var qlBase = 64 * h;
                var qhBase = 32 * h;
                var scBase = 8 * h;
                var dstBase = 128 * h;

                for (var l = 0; l < 32; l++)
                {
                    var qhByte = qh[qhBase + l];
                    var is_ = l / 16;     // 0 or 1 within the 32-element stride

                    var q1 = ql[qlBase + l] & 0x0F | (((qhByte >> 0) & 0x03) << 4);
                    var q2 = ql[qlBase + l + 32] & 0x0F | (((qhByte >> 2) & 0x03) << 4);
                    var q3 = ql[qlBase + l] >> 4 | (((qhByte >> 4) & 0x03) << 4);
                    var q4 = ql[qlBase + l + 32] >> 4 | (((qhByte >> 6) & 0x03) << 4);

                    // scales are int8 — sign-extend
                    var s1 = (sbyte)sc[scBase + is_ + 0];
                    var s2 = (sbyte)sc[scBase + is_ + 2];
                    var s3 = (sbyte)sc[scBase + is_ + 4];
                    var s4 = (sbyte)sc[scBase + is_ + 6];

                    dst256[dstBase + l] = d * s1 * (q1 - 32);
                    dst256[dstBase + l + 32] = d * s2 * (q2 - 32);
                    dst256[dstBase + l + 64] = d * s3 * (q3 - 32);
                    dst256[dstBase + l + 96] = d * s4 * (q4 - 32);
                }
            }
        }
    }
}
