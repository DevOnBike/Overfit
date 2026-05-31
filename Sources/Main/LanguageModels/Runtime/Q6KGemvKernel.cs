// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Runtime;

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

        /// <summary>
        /// AVX2 8×8 Q6_K GEMV — vectorised over the 8 output rows (lanes), no per-row
        /// horizontal reduction. Same math as <see cref="GemvScalar"/>: forms each unsigned
        /// 6-bit weight (ql nibble | qh 2-bit &lt;&lt; 4), accumulates Σ q·a per row, applies the
        /// −32 bias via Σa, scales (signed int8), and the per-super-block d·activationScale.
        /// Validated bit-close to <see cref="GemvScalar"/>.
        /// </summary>
        /// <summary>
        /// Opt-in flag (shared with <see cref="Q4KGemvKernel.Enabled"/> via
        /// <c>OVERFIT_REPACK_GEMV</c>) for the repacked 8×8 decode GEMV. AVX2-only.
        /// </summary>
        public static bool Enabled => Q4KGemvKernel.Enabled;

        public static unsafe void GemvAvx2(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output)
        {
            var nb = inputSize / 256;
            fixed (byte* rep = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (float* o = output)
            {
                ComputeGroupRange(rep, aq, asc, o, 0, outputSize / 8, nb);
            }
        }

        /// <summary>
        /// Parallel full-matrix GEMV — output-row-groups split across the decode dispatch
        /// (capped / spin-pool). The <c>block_q6_Kx8</c> layout is row-group-major, so each
        /// worker owns a disjoint contiguous slice of weights + outputs.
        /// </summary>
        public static unsafe void GemvParallel(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            Span<float> output)
        {
            var nb = inputSize / 256;
            var groups = outputSize / 8;
            fixed (byte* rep = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (float* o = output)
            {
                var ctx = new GemvContext { Rep = rep, Aq = aq, Asc = asc, Out = o, Nb = nb };
                OverfitParallelFor.ForDecode(0, groups, &GroupChunk, &ctx);
            }
        }

        private unsafe struct GemvContext
        {
            public byte* Rep;
            public sbyte* Aq;
            public float* Asc;
            public float* Out;
            public int Nb;
        }

        private static unsafe void GroupChunk(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<GemvContext>(context);
            ComputeGroupRange(ctx.Rep, ctx.Aq, ctx.Asc, ctx.Out, start, end, ctx.Nb);
        }

        /// <summary>Output-row-groups [<paramref name="start"/>, <paramref name="end"/>) — the maddubs core.</summary>
        private static unsafe void ComputeGroupRange(
            byte* rep, sbyte* aqAll, float* asc, float* outp, int start, int end, int nb)
        {
            var m4b = Vector256.Create((byte)0x0F);
            var m2 = Vector256.Create((byte)0x03);
            var m32 = Vector256.Create((byte)32);
            var ones = Vector256.Create((short)1);
            var reduce = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

            {
                for (var x = start; x < end; x++)
                {
                    var bptr = rep + (long)x * nb * BlockKx8Bytes;
                    var sumf = Vector256<float>.Zero;

                    for (var l = 0; l < nb; l++)
                    {
                        var blk = bptr + (long)l * BlockKx8Bytes;
                        var scales = blk + DstScalesOffset;
                        var ql = blk + DstQlOffset;
                        var qh = blk + DstQhOffset;
                        var dVec = LoadF16x8Int(blk);
                        var aD = asc[l];
                        var aqs = aqAll + l * 256;

                        var iacc = Vector256<int>.Zero;

                        for (var k = 0; k < 16; k++)
                        {
                            var baseL = (k / 8) * 128 + (k % 8) * 8;
                            var baseH = baseL + 64;
                            var qhShiftL = (byte)(((baseL % 128) / 32) * 2);
                            var qhShiftH = (byte)(((baseH % 128) / 32) * 2);
                            var qhHalfL = (baseL / 128) * 32;
                            var qhHalfH = (baseH / 128) * 32;
                            var qhBlockL = ((qhHalfL + (baseL % 32)) / 8) * 64;
                            var qhBlockH = ((qhHalfH + (baseH % 32)) / 8) * 64;

                            // Row-major loads: rows 0-3 in *03, rows 4-7 in *47 (8 bytes/row).
                            var ql03 = Vector256.Load(ql + k * 64);
                            var ql47 = Vector256.Load(ql + k * 64 + 32);
                            var qhL03 = Vector256.Load(qh + qhBlockL);
                            var qhL47 = Vector256.Load(qh + qhBlockL + 32);
                            var qhH03 = Vector256.Load(qh + qhBlockH);
                            var qhH47 = Vector256.Load(qh + qhBlockH + 32);

                            // L stream = low nibble | (qhL 2-bit << 4); H stream = high nibble | (qhH 2-bit << 4).
                            var qLu03 = Avx2.Or(LoNib(ql03, m4b), QhBits(qhL03, qhShiftL, m2));
                            var qLu47 = Avx2.Or(LoNib(ql47, m4b), QhBits(qhL47, qhShiftL, m2));
                            var qHu03 = Avx2.Or(HiNib(ql03, m4b), QhBits(qhH03, qhShiftH, m2));
                            var qHu47 = Avx2.Or(HiNib(ql47, m4b), QhBits(qhH47, qhShiftH, m2));

                            // Activations tiled so both rows in each 128-lane see a[base..base+8].
                            var actL = TileAct(aqs + baseL);
                            var actH = TileAct(aqs + baseH);

                            // maddubs (32 elems/op), fold −32 via maddubs(32, act), reduce to 8 row-lanes.
                            var sumL = ReduceRows(
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qLu03, actL), Avx2.MultiplyAddAdjacent(m32, actL)),
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qLu47, actL), Avx2.MultiplyAddAdjacent(m32, actL)),
                                ones, reduce);
                            var sumH = ReduceRows(
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qHu03, actH), Avx2.MultiplyAddAdjacent(m32, actH)),
                                Avx2.Subtract(Avx2.MultiplyAddAdjacent(qHu47, actH), Avx2.MultiplyAddAdjacent(m32, actH)),
                                ones, reduce);

                            var scaleL = ScaleVec(scales + (baseL / 16) * 8);
                            var scaleH = ScaleVec(scales + (baseH / 16) * 8);

                            iacc = Avx2.Add(iacc, Avx2.Add(
                                Avx2.MultiplyLow(sumL, scaleL), Avx2.MultiplyLow(sumH, scaleH)));
                        }

                        sumf = Fma.MultiplyAdd(
                            Avx.ConvertToVector256Single(iacc), Avx.Multiply(dVec, Vector256.Create(aD)), sumf);
                    }

                    sumf.Store(outp + x * 8);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> LoNib(Vector256<byte> v, Vector256<byte> m4b) => Avx2.And(v, m4b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> HiNib(Vector256<byte> v, Vector256<byte> m4b) =>
            Avx2.And(Avx2.ShiftRightLogical(v.AsInt16(), 4).AsByte(), m4b);

        // 2-bit qh field at `shift`, masked, shifted into the high-nibble position (<<4).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> QhBits(Vector256<byte> qh, byte shift, Vector256<byte> m2)
        {
            var bits = Avx2.And(Avx2.ShiftRightLogical(qh.AsInt16(), shift).AsByte(), m2);
            return Avx2.ShiftLeftLogical(bits.AsInt16(), 4).AsByte();
        }

        // Broadcast 8 activation bytes to [a0..a7, a0..a7] in BOTH 128-bit lanes.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<sbyte> TileAct(sbyte* a)
        {
            var bits = Unsafe.ReadUnaligned<long>(a);
            var v = Vector128.Create(bits, bits).AsSByte();
            return Vector256.Create(v, v);
        }

        // p03 = [r0:4|r1:4|r2:4|r3:4] int16 pair-sums, p47 likewise → 8 int32 lanes [r0..r7].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> ReduceRows(
            Vector256<short> p03, Vector256<short> p47, Vector256<short> ones, Vector256<int> reduce)
        {
            var m03 = Avx2.MultiplyAddAdjacent(p03, ones); // [r0a,r0b,r1a,r1b|r2a,r2b,r3a,r3b]
            var m47 = Avx2.MultiplyAddAdjacent(p47, ones);
            var h = Avx2.HorizontalAdd(m03, m47);          // [r0,r1,r4,r5,r2,r3,r6,r7]
            return Avx2.PermuteVar8x32(h, reduce);         // [r0,r1,r2,r3,r4,r5,r6,r7]
        }

        // 8 lanes of SIGNED int8 scales at [base..base+8].
        private static unsafe Vector256<int> ScaleVec(byte* sc) =>
            Avx2.ConvertToVector256Int32(Vector128.CreateScalar(Unsafe.ReadUnaligned<long>(sc)).AsSByte());

        private static unsafe Vector256<float> LoadF16x8Int(byte* blk)
        {
            var u = (ushort*)blk;
            return Vector256.Create(
                (float)BitConverter.UInt16BitsToHalf(u[0]), (float)BitConverter.UInt16BitsToHalf(u[1]),
                (float)BitConverter.UInt16BitsToHalf(u[2]), (float)BitConverter.UInt16BitsToHalf(u[3]),
                (float)BitConverter.UInt16BitsToHalf(u[4]), (float)BitConverter.UInt16BitsToHalf(u[5]),
                (float)BitConverter.UInt16BitsToHalf(u[6]), (float)BitConverter.UInt16BitsToHalf(u[7]));
        }
    }
}
