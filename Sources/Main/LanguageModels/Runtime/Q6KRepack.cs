// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Repacks standard Q6_K weights into llama.cpp's <c>block_q6_Kx8</c> layout (8 output
    /// rows interleaved per super-block column) — the foundation for the repacked 8×8 Q6_K
    /// decode GEMV. Faithful port of llama.cpp <c>make_block_q6_Kx8</c> /
    /// <c>repack_q6_K_to_q6_K_8_bl</c> (ggml-cpu/repack.cpp).
    ///
    /// <para>One <c>block_q6_Kx8</c> is 1680 bytes (== 8 × 210) holding, for one super-block
    /// column, the 8 rows' data:</para>
    /// <list type="bullet">
    ///   <item><c>d[8]</c>        — 8 × fp16 super-block scale (bytes 0..15)</item>
    ///   <item><c>scales[128]</c> — 16 int8 sub-block scales × 8 rows, transposed (bytes 16..143)</item>
    ///   <item><c>ql[1024]</c>    — low 4 bits, interleaved 8 bytes at a time (bytes 144..1167)</item>
    ///   <item><c>qh[512]</c>     — high 2 bits, interleaved 8 bytes at a time (bytes 1168..1679)</item>
    /// </list>
    ///
    /// <para>Input is the standard Q6_K super-block (210 bytes: <c>ql</c>:128, <c>qh</c>:64,
    /// <c>scales</c>:16 int8, <c>d</c>:2 fp16) — the layout <see cref="Q6KWeight"/> stores.</para>
    /// </summary>
    public static class Q6KRepack
    {
        /// <summary>Bytes per standard Q6_K super-block (== <see cref="Q6KWeight.SuperBlockBytes"/>).</summary>
        public const int SuperBlockBytes = 210;

        /// <summary>Output rows interleaved per <c>block_q6_Kx8</c>.</summary>
        public const int RowsInterleaved = 8;

        /// <summary>Bytes per <c>block_q6_Kx8</c> (== 8 × 210).</summary>
        public const int BlockKx8Bytes = 1680;

        // Standard Q6_K (210 B): ql[0..128], qh[128..192], scales[192..208], d[208..210].
        private const int SrcQlOffset = 0;
        private const int SrcQhOffset = 128;
        private const int SrcScalesOffset = 192;
        private const int SrcDOffset = 208;

        // block_q6_Kx8 (1680 B): d[0..16], scales[16..144], ql[144..1168], qh[1168..1680].
        private const int DstDOffset = 0;
        private const int DstScalesOffset = 16;
        private const int DstQlOffset = 144;
        private const int DstQhOffset = 1168;

        /// <summary>
        /// Repacks a full output-major Q6_K weight matrix into the <c>block_q6_Kx8</c> layout.
        /// Requires <c>outputSize % 8 == 0</c> and <c>inputSize % 256 == 0</c>.
        /// </summary>
        public static byte[] RepackMatrix(ReadOnlySpan<byte> q6k, int outputSize, int inputSize)
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
            if (q6k.Length < expected)
            {
                throw new ArgumentException($"q6k span ({q6k.Length}) smaller than expected {expected} bytes.", nameof(q6k));
            }

            var dst = new byte[(long)(outputSize / RowsInterleaved) * superBlocksPerRow * BlockKx8Bytes];
            Span<byte> gather = stackalloc byte[RowsInterleaved * SuperBlockBytes];

            var dstOffset = 0;
            for (var rowGroup = 0; rowGroup < outputSize; rowGroup += RowsInterleaved)
            {
                for (var col = 0; col < superBlocksPerRow; col++)
                {
                    for (var i = 0; i < RowsInterleaved; i++)
                    {
                        var srcSb = (long)(rowGroup + i) * superBlocksPerRow + col;
                        q6k.Slice((int)(srcSb * SuperBlockBytes), SuperBlockBytes)
                           .CopyTo(gather.Slice(i * SuperBlockBytes, SuperBlockBytes));
                    }

                    MakeBlockQ6Kx8(gather, dst.AsSpan(dstOffset, BlockKx8Bytes));
                    dstOffset += BlockKx8Bytes;
                }
            }

            return dst;
        }

        /// <summary>Interleaves 8 standard Q6_K super-blocks into one <c>block_q6_Kx8</c>.</summary>
        public static void MakeBlockQ6Kx8(ReadOnlySpan<byte> src8, Span<byte> dst)
        {
            // d[8] — copy the fp16 super-block scale of each of the 8 rows.
            for (var i = 0; i < 8; i++)
            {
                src8.Slice(i * SuperBlockBytes + SrcDOffset, 2).CopyTo(dst.Slice(DstDOffset + i * 2, 2));
            }

            // ql[1024] — interleave low bits 8 bytes at a time. end_ls = 256*4/8 = 128.
            for (var i = 0; i < 128; i++)
            {
                var srcId = i % 8;
                var srcOff = (i / 8) * 8;
                var dstOff = i * 8;
                src8.Slice(srcId * SuperBlockBytes + SrcQlOffset + srcOff, 8)
                    .CopyTo(dst.Slice(DstQlOffset + dstOff, 8));
            }

            // qh[512] — interleave high bits 8 bytes at a time. end_hs = 128/2 = 64.
            for (var i = 0; i < 64; i++)
            {
                var srcId = i % 8;
                var srcOff = (i / 8) * 8;
                var dstOff = i * 8;
                src8.Slice(srcId * SuperBlockBytes + SrcQhOffset + srcOff, 8)
                    .CopyTo(dst.Slice(DstQhOffset + dstOff, 8));
            }

            // scales[128] — out.scales[j*8 + i] = in[i].scales[j] (16 sub-blocks × 8 rows).
            for (var i = 0; i < 8; i++)
            {
                for (var j = 0; j < 16; j++)
                {
                    dst[DstScalesOffset + j * 8 + i] = src8[i * SuperBlockBytes + SrcScalesOffset + j];
                }
            }
        }
    }
}
