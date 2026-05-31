// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Repacks standard row-major Q4_K weights into llama.cpp's <c>block_q4_Kx8</c> layout
    /// (8 output rows interleaved per super-block column) — the foundation for the
    /// repacked 8×8 GEMV decode kernel, where 8 output rows are produced together in SIMD
    /// lanes with no per-row horizontal reduction.
    ///
    /// <para>Faithful port of llama.cpp <c>make_block_q4_Kx8</c> / <c>repack_q4_K_to_q4_K_8_bl</c>
    /// (ggml-cpu/repack.cpp). One <c>block_q4_Kx8</c> is 1152 bytes (== 8 × 144) and holds, for
    /// one super-block column, the 8 rows' data:</para>
    /// <list type="bullet">
    ///   <item><c>d[8]</c>     — 8 × fp16 super-block scale (bytes 0..15)</item>
    ///   <item><c>dmin[8]</c>  — 8 × fp16 super-block min-scale (bytes 16..31)</item>
    ///   <item><c>scales[96]</c> — the 8 rows' 6-bit scales/mins repacked (bytes 32..127)</item>
    ///   <item><c>qs[1024]</c> — the 8 rows' 4-bit quants, interleaved 8 bytes at a time (bytes 128..1151)</item>
    /// </list>
    ///
    /// <para>Input is the standard Q4_K super-block (144 bytes: <c>d</c>:2, <c>dmin</c>:2,
    /// <c>scales</c>:12, <c>qs</c>:128) — the same layout <see cref="Q4KWeight"/> stores.</para>
    /// </summary>
    public static class Q4KRepack
    {
        /// <summary>Bytes per standard Q4_K super-block (== <see cref="Q4KWeight.SuperBlockBytes"/>).</summary>
        public const int SuperBlockBytes = 144;

        /// <summary>Output rows interleaved per <c>block_q4_Kx8</c>.</summary>
        public const int RowsInterleaved = 8;

        /// <summary>Bytes per <c>block_q4_Kx8</c> (== 8 × 144).</summary>
        public const int BlockKx8Bytes = 1152;

        // Byte offsets inside a standard 144-byte Q4_K super-block.
        private const int SrcDOffset = 0;       // fp16 d
        private const int SrcDminOffset = 2;    // fp16 dmin
        private const int SrcScalesOffset = 4;  // 12 bytes of 6-bit scales/mins
        private const int SrcQsOffset = 16;     // 128 bytes of 4-bit quants

        // Byte offsets inside a 1152-byte block_q4_Kx8.
        private const int DstDOffset = 0;        // 8 × fp16
        private const int DstDminOffset = 16;    // 8 × fp16
        private const int DstScalesOffset = 32;  // 96 bytes
        private const int DstQsOffset = 128;     // 1024 bytes

        /// <summary>
        /// Repacks a full output-major Q4_K weight matrix (<paramref name="outputSize"/> rows ×
        /// <paramref name="inputSize"/> cols, row-major super-blocks) into the
        /// <c>block_q4_Kx8</c> layout. Requires <c>outputSize % 8 == 0</c> and
        /// <c>inputSize % 256 == 0</c>. The repacked stream is, per group of 8 rows, one
        /// <c>block_q4_Kx8</c> per super-block column, in column order — exactly the order the
        /// 8×8 GEMV consumes.
        /// </summary>
        public static byte[] RepackMatrix(ReadOnlySpan<byte> q4k, int outputSize, int inputSize)
        {
            if (outputSize % RowsInterleaved != 0)
            {
                throw new ArgumentException($"outputSize ({outputSize}) must be a multiple of {RowsInterleaved}.", nameof(outputSize));
            }
            if (inputSize % 256 != 0)
            {
                throw new ArgumentException($"inputSize ({inputSize}) must be a multiple of 256.", nameof(inputSize));
            }

            var superBlocksPerRow = inputSize / 256;
            var expected = (long)outputSize * superBlocksPerRow * SuperBlockBytes;
            if (q4k.Length < expected)
            {
                throw new ArgumentException($"q4k span ({q4k.Length}) smaller than expected {expected} bytes.", nameof(q4k));
            }

            var dst = new byte[(long)(outputSize / RowsInterleaved) * superBlocksPerRow * BlockKx8Bytes];
            Span<byte> gather = stackalloc byte[RowsInterleaved * SuperBlockBytes];

            var dstOffset = 0;
            for (var rowGroup = 0; rowGroup < outputSize; rowGroup += RowsInterleaved)
            {
                for (var col = 0; col < superBlocksPerRow; col++)
                {
                    // Gather the 8 rows' super-block at this column: src[col + i*superBlocksPerRow].
                    for (var i = 0; i < RowsInterleaved; i++)
                    {
                        var srcSb = (long)(rowGroup + i) * superBlocksPerRow + col;
                        q4k.Slice((int)(srcSb * SuperBlockBytes), SuperBlockBytes)
                           .CopyTo(gather.Slice(i * SuperBlockBytes, SuperBlockBytes));
                    }

                    MakeBlockQ4Kx8(gather, dst.AsSpan(dstOffset, BlockKx8Bytes));
                    dstOffset += BlockKx8Bytes;
                }
            }

            return dst;
        }

        /// <summary>
        /// Interleaves 8 standard Q4_K super-blocks (one per row, same column) into one
        /// <c>block_q4_Kx8</c>. Direct port of llama.cpp <c>make_block_q4_Kx8</c> with
        /// <c>blck_size_interleave = 8</c>.
        /// </summary>
        public static void MakeBlockQ4Kx8(ReadOnlySpan<byte> src8, Span<byte> dst)
        {
            // d[8], dmin[8] — copy the fp16 scale/min of each of the 8 rows.
            for (var i = 0; i < 8; i++)
            {
                var rowBase = i * SuperBlockBytes;
                src8.Slice(rowBase + SrcDOffset, 2).CopyTo(dst.Slice(DstDOffset + i * 2, 2));
                src8.Slice(rowBase + SrcDminOffset, 2).CopyTo(dst.Slice(DstDminOffset + i * 2, 2));
            }

            // qs[1024] — interleave the quants 8 bytes at a time: out chunk i takes 8 bytes
            // from row (i % 8) at its source chunk (i / 8). end = 256*4/8 = 128 chunks.
            const int end = 256 * 4 / 8;
            for (var i = 0; i < end; i++)
            {
                var srcId = i % 8;
                var srcOffset = (i / 8) * 8;
                var dstOffset = i * 8;
                src8.Slice(srcId * SuperBlockBytes + SrcQsOffset + srcOffset, 8)
                    .CopyTo(dst.Slice(DstQsOffset + dstOffset, 8));
            }

            // scales[96] — unpack the 6-bit scales/mins of the 8 rows and repack so each
            // 12-byte group holds all 8 rows' (scale, min) for one sub-block.
            Span<byte> s = stackalloc byte[8];
            Span<byte> m = stackalloc byte[8];

            for (var i = 0; i < 4; i++)
            {
                for (var j = 0; j < 8; j++)
                {
                    var sc = src8.Slice(j * SuperBlockBytes + SrcScalesOffset, 12);
                    s[j] = (byte)(sc[i] & 63);
                    m[j] = (byte)(sc[i + 4] & 63);
                }

                var o = DstScalesOffset + i * 12;
                dst[o] = (byte)((s[0] & 63) + ((s[4] & 48) << 2));
                dst[o + 1] = (byte)((s[1] & 63) + ((s[5] & 48) << 2));
                dst[o + 2] = (byte)((s[2] & 63) + ((s[6] & 48) << 2));
                dst[o + 3] = (byte)((s[3] & 63) + ((s[7] & 48) << 2));
                dst[o + 4] = (byte)((m[0] & 63) + ((m[4] & 48) << 2));
                dst[o + 5] = (byte)((m[1] & 63) + ((m[5] & 48) << 2));
                dst[o + 6] = (byte)((m[2] & 63) + ((m[6] & 48) << 2));
                dst[o + 7] = (byte)((m[3] & 63) + ((m[7] & 48) << 2));
                dst[o + 8] = (byte)((s[4] & 15) + ((m[4] & 15) << 4));
                dst[o + 9] = (byte)((s[5] & 15) + ((m[5] & 15) << 4));
                dst[o + 10] = (byte)((s[6] & 15) + ((m[6] & 15) << 4));
                dst[o + 11] = (byte)((s[7] & 15) + ((m[7] & 15) << 4));
            }

            for (var i = 0; i < 4; i++)
            {
                for (var j = 0; j < 8; j++)
                {
                    var sc = src8.Slice(j * SuperBlockBytes + SrcScalesOffset, 12);
                    s[j] = (byte)(((sc[i] & 192) >> 2) | (sc[i + 8] & 15));
                    m[j] = (byte)(((sc[i + 4] & 192) >> 2) | ((sc[i + 8] & 240) >> 4));
                }

                var o = DstScalesOffset + i * 12 + 48;
                dst[o] = (byte)((s[0] & 63) + ((s[4] & 48) << 2));
                dst[o + 1] = (byte)((s[1] & 63) + ((s[5] & 48) << 2));
                dst[o + 2] = (byte)((s[2] & 63) + ((s[6] & 48) << 2));
                dst[o + 3] = (byte)((s[3] & 63) + ((s[7] & 48) << 2));
                dst[o + 4] = (byte)((m[0] & 63) + ((m[4] & 48) << 2));
                dst[o + 5] = (byte)((m[1] & 63) + ((m[5] & 48) << 2));
                dst[o + 6] = (byte)((m[2] & 63) + ((m[6] & 48) << 2));
                dst[o + 7] = (byte)((m[3] & 63) + ((m[7] & 48) << 2));
                dst[o + 8] = (byte)((s[4] & 15) + ((m[4] & 15) << 4));
                dst[o + 9] = (byte)((s[5] & 15) + ((m[5] & 15) << 4));
                dst[o + 10] = (byte)((s[6] & 15) + ((m[6] & 15) << 4));
                dst[o + 11] = (byte)((s[7] & 15) + ((m[7] & 15) << 4));
            }
        }
    }
}
