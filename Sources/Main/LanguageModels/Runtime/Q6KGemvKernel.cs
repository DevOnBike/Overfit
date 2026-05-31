// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Repacked 8×8 Q6_K GEMV — 8 output rows from a <see cref="Q6KRepack"/> <c>block_q6_Kx8</c>
    /// weight and a Q8_K activation. llama.cpp has NO x86 AVX2 Q6_K 8×8 kernel (only ARM NEON +
    /// a generic scalar), so this is a NOVEL kernel: <see cref="GemvScalar"/> is the faithful port
    /// of the generic (the correctness oracle); the AVX2 path is built against it.
    /// </summary>
    public static class Q6KGemvKernel
    {
        private const int BlockKx8Bytes = Q6KRepack.BlockKx8Bytes; // 1680
        private const int DstScalesOffset = 16;
        private const int DstQlOffset = 144;
        private const int DstQhOffset = 1168;

        /// <summary>
        /// Scalar reference GEMV over the repacked layout — port of llama.cpp
        /// <c>ggml_gemv_q6_K_NxM_q8_K_generic_impl&lt;8,8&gt;</c>. Q6_K applies the <c>−32</c>
        /// bias per element (no bsums term). The correctness oracle for the AVX2 path.
        /// </summary>
        public static void GemvScalar(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output)
        {
            var nb = inputSize / 256;
            Span<float> sumf = stackalloc float[8];

            for (var x = 0; x < outputSize / 8; x++)
            {
                var bptr = (long)x * nb * BlockKx8Bytes;
                sumf.Clear();

                for (var l = 0; l < nb; l++)
                {
                    var blk = repacked.Slice((int)(bptr + (long)l * BlockKx8Bytes), BlockKx8Bytes);
                    var scales = blk.Slice(DstScalesOffset, 128);
                    var ql = blk.Slice(DstQlOffset, 1024);
                    var qh = blk.Slice(DstQhOffset, 512);

                    var aD = actScales[l];
                    var aqs = actQuants.Slice(l * 256, 256);

                    for (var k = 0; k < 16; k++)
                    {
                        var baseL = (k / 8) * 128 + (k % 8) * 8;
                        var baseH = baseL + 64;
                        var scaleIdxL = baseL / 16;
                        var scaleIdxH = baseH / 16;
                        var qhShiftL = ((baseL % 128) / 32) * 2;
                        var qhShiftH = ((baseH % 128) / 32) * 2;
                        var qhHalfL = (baseL / 128) * 32;
                        var qhHalfH = (baseH / 128) * 32;

                        for (var j = 0; j < 8; j++)
                        {
                            int scaleL = (sbyte)scales[scaleIdxL * 8 + j];
                            int scaleH = (sbyte)scales[scaleIdxH * 8 + j];

                            var sumiL = 0;
                            var sumiH = 0;

                            for (var i = 0; i < 8; i++)
                            {
                                var qlPos = k * 64 + j * 8 + i;
                                var l4 = ql[qlPos] & 0xF;
                                var hi4 = (ql[qlPos] >> 4) & 0xF;

                                var qhIdxL = qhHalfL + ((baseL + i) % 32);
                                var qhOffL = (qhIdxL / 8) * 64 + j * 8 + (qhIdxL % 8);
                                var hi2L = (qh[qhOffL] >> qhShiftL) & 0x3;

                                var qhIdxH = qhHalfH + ((baseH + i) % 32);
                                var qhOffH = (qhIdxH / 8) * 64 + j * 8 + (qhIdxH % 8);
                                var hi2H = (qh[qhOffH] >> qhShiftH) & 0x3;

                                var qL = ((hi2L << 4) | l4) - 32;
                                var qH = ((hi2H << 4) | hi4) - 32;

                                sumiL += qL * aqs[baseL + i];
                                sumiH += qH * aqs[baseH + i];
                            }

                            var d = (float)BitConverter.UInt16BitsToHalf(
                                MemoryMarshal.Read<ushort>(blk.Slice(j * 2, 2)));
                            sumf[j] += (sumiL * scaleL + sumiH * scaleH) * d * aD;
                        }
                    }
                }

                for (var j = 0; j < 8; j++)
                {
                    output[x * 8 + j] = sumf[j];
                }
            }
        }
    }
}
