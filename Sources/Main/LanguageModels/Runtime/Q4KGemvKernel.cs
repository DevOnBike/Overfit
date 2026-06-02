// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Repacked 8×8 Q4_K GEMV — produces 8 output rows at once from a
    /// <see cref="Q4KRepack"/> <c>block_q4_Kx8</c> weight and a Q8_K activation, with the 8
    /// row dot-products accumulated in SIMD lanes (no per-row horizontal reduction).
    /// Faithful AVX2 port of llama.cpp <c>ggml_gemv_q4_K_8x8_q8_K</c> (arch/x86/repack.cpp).
    /// Validated bit-close (maxRelDiff ~1e-6) to <see cref="Q4KDotKernel"/>; ~2× faster
    /// per core, &gt;3× parallelised.
    /// </summary>
    public static unsafe class Q4KGemvKernel
    {
        private const int BlockKx8Bytes = Q4KRepack.BlockKx8Bytes; // 1152
        private const int DstScalesOffset = 32;
        private const int DstQsOffset = 128;

        /// <summary>
        /// Opt-in (<c>OVERFIT_REPACK_GEMV=1</c>) for the repacked 8×8 decode GEMV. Off by
        /// default — it allocates a repacked weight copy per Q4_K FFN tensor (adds RAM) and is
        /// AVX2-only. When on, the decode FFN gate/up projections route here.
        /// </summary>
        public static readonly bool Enabled = ResolveEnabled();

        private static bool ResolveEnabled()
        {
            if (!CpuFeatures.HasAvx2)
            {
                return false;
            }

            var raw = Environment.GetEnvironmentVariable("OVERFIT_REPACK_GEMV");
            return raw is "1" || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Sequential full-matrix GEMV (one thread).</summary>
        public static void Gemv(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            ReadOnlySpan<short> actBsums,
            Span<float> output)
        {
            var nb = inputSize / 256;
            fixed (byte* w = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (short* ab = actBsums)
            fixed (float* o = output)
            {
                for (var x = 0; x < outputSize / 8; x++)
                {
                    ComputeGroup(w, aq, asc, ab, o, x, nb);
                }
            }
        }

        /// <summary>
        /// Parallel full-matrix GEMV — the output-row-groups are split across the decode
        /// dispatch (capped / spin-pool). The <c>block_q4_Kx8</c> layout is row-group-major,
        /// so each worker owns a disjoint contiguous slice of weights + outputs.
        /// </summary>
        public static void GemvParallel(
            ReadOnlySpan<byte> repacked,
            int outputSize,
            int inputSize,
            ReadOnlySpan<sbyte> actQuants,
            ReadOnlySpan<float> actScales,
            ReadOnlySpan<short> actBsums,
            Span<float> output)
        {
            var nb = inputSize / 256;
            var groups = outputSize / 8;

            fixed (byte* w = repacked)
            fixed (sbyte* aq = actQuants)
            fixed (float* asc = actScales)
            fixed (short* ab = actBsums)
            fixed (float* o = output)
            {
                var ctx = new GemvContext
                {
                    W = w,
                    Aq = aq,
                    Asc = asc,
                    Ab = ab,
                    Output = o,
                    Nb = nb,
                };
                OverfitParallelFor.ForDecode(0, groups, &GroupChunk, &ctx);
            }
        }

        private struct GemvContext
        {
            public byte* W;
            public sbyte* Aq;
            public float* Asc;
            public short* Ab;
            public float* Output;
            public int Nb;
        }

        private static void GroupChunk(int start, int end, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<GemvContext>(context);
            for (var x = start; x < end; x++)
            {
                ComputeGroup(ctx.W, ctx.Aq, ctx.Asc, ctx.Ab, ctx.Output, x, ctx.Nb);
            }
        }

        /// <summary>One row-group (8 output rows) — the validated inner kernel.</summary>
        private static void ComputeGroup(byte* w, sbyte* aq, float* asc, short* ab, float* output, int x, int nb)
        {
            var m4b = Vector256.Create((byte)0x0F);
            var deltamask = Vector128.Create((byte)0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
            var scalemask = Vector128.Create((byte)0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
            var finalpermute = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);
            const uint kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;

            var bptr = w + (long)x * nb * BlockKx8Bytes;
            var accRow = Vector256<float>.Zero;
            var accMin = Vector256<float>.Zero;

            // Scratch for the unpacked 6-bit scales/mins, hoisted out of BOTH loops (CA2014:
            // stackalloc in a loop grows the frame each iteration). Fully overwritten per use.
            var u0 = stackalloc uint[4];
            var u1 = stackalloc uint[4];

            for (var b = 0; b < nb; b++)
            {
                var blk = bptr + (long)b * BlockKx8Bytes;

                var rowScale = Vector256.Create(asc[b]);
                var colScale = LoadF16x8Rearrange(blk, deltamask);
                var colDmin = LoadF16x8(blk + 16);

                var q8sums = Vector256.Load(ab + b * 16);
                var q8sHadd = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                var q8s = Vector256.Create(q8sHadd, q8sHadd).AsInt16();

                var iaccB = Vector256<int>.Zero;
                var iaccMinB = Vector256<int>.Zero;

                var qsBase = blk + DstQsOffset;
                var scBase = blk + DstScalesOffset;

                for (var sb = 0; sb < 4; sb++)
                {
                    var qs = qsBase + sb * 256;
                    var raw0123_0 = Vector256.Load(qs);
                    var raw4567_0 = Vector256.Load(qs + 32);
                    var raw0123_1 = Vector256.Load(qs + 64);
                    var raw4567_1 = Vector256.Load(qs + 96);
                    var raw0123_2 = Vector256.Load(qs + 128);
                    var raw4567_2 = Vector256.Load(qs + 160);
                    var raw0123_3 = Vector256.Load(qs + 192);
                    var raw4567_3 = Vector256.Load(qs + 224);

                    var v0123_00 = Avx2.And(raw0123_0, m4b);
                    var v4567_00 = Avx2.And(raw4567_0, m4b);
                    var v0123_01 = Avx2.And(raw0123_1, m4b);
                    var v4567_01 = Avx2.And(raw4567_1, m4b);
                    var v0123_02 = Avx2.And(raw0123_2, m4b);
                    var v4567_02 = Avx2.And(raw4567_2, m4b);
                    var v0123_03 = Avx2.And(raw0123_3, m4b);
                    var v4567_03 = Avx2.And(raw4567_3, m4b);

                    var v0123_10 = Avx2.And(Hi(raw0123_0), m4b);
                    var v4567_10 = Avx2.And(Hi(raw4567_0), m4b);
                    var v0123_11 = Avx2.And(Hi(raw0123_1), m4b);
                    var v4567_11 = Avx2.And(Hi(raw4567_1), m4b);
                    var v0123_12 = Avx2.And(Hi(raw0123_2), m4b);
                    var v4567_12 = Avx2.And(Hi(raw4567_2), m4b);
                    var v0123_13 = Avx2.And(Hi(raw0123_3), m4b);
                    var v4567_13 = Avx2.And(Hi(raw4567_3), m4b);

                    u0[0] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb);
                    u0[1] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 4);
                    u0[2] = Unsafe.ReadUnaligned<uint>(scBase + 24 * sb + 8);
                    u1[0] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24);
                    u1[1] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 4);
                    u1[2] = Unsafe.ReadUnaligned<uint>(scBase + 12 + sb * 24 + 8);
                    Unpack(u0, kmask1, kmask2, kmask3);
                    Unpack(u1, kmask1, kmask2, kmask3);

                    var ms0 = Vector128.Create(u0[0], u0[1], u0[2], u0[3]).AsByte();
                    var ms1 = Vector128.Create(u1[0], u1[1], u1[2], u1[3]).AsByte();
                    var scales0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms0, scalemask));
                    var scales1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(ms1, scalemask));
                    var mins01 = Avx2.ConvertToVector256Int16(
                        Sse2.UnpackLow(
                            Sse2.Shuffle(ms0.AsInt32(), 78).AsByte(),
                            Sse2.Shuffle(ms1.AsInt32(), 78).AsByte()));

                    var aqb = aq + b * 256 + sb * 64;
                    var l00 = BroadcastLo(aqb);
                    var l01 = BroadcastLo(aqb + 16);
                    var l10 = BroadcastLo(aqb + 32);
                    var l11 = BroadcastLo(aqb + 48);

                    var iacc0 = Vector256<short>.Zero;
                    var iacc1 = Vector256<short>.Zero;

                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_00, Sh(v4567_00, 177)), Sh32(l00, 0)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_00, 177), v4567_00), Sh32(l00, 85)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_01, Sh(v4567_01, 177)), Sh32(l00, 170)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_01, 177), v4567_01), Sh32(l00, 255)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_02, Sh(v4567_02, 177)), Sh32(l01, 0)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_02, 177), v4567_02), Sh32(l01, 85)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(v0123_03, Sh(v4567_03, 177)), Sh32(l01, 170)));
                    iacc0 = Avx2.Add(iacc0, Mul(Blend(Sh(v0123_03, 177), v4567_03), Sh32(l01, 255)));
                    var iacc0i = Avx2.MultiplyAddAdjacent(iacc0, scales0);

                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_10, Sh(v4567_10, 177)), Sh32(l10, 0)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_10, 177), v4567_10), Sh32(l10, 85)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_11, Sh(v4567_11, 177)), Sh32(l10, 170)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_11, 177), v4567_11), Sh32(l10, 255)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_12, Sh(v4567_12, 177)), Sh32(l11, 0)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_12, 177), v4567_12), Sh32(l11, 85)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(v0123_13, Sh(v4567_13, 177)), Sh32(l11, 170)));
                    iacc1 = Avx2.Add(iacc1, Mul(Blend(Sh(v0123_13, 177), v4567_13), Sh32(l11, 255)));
                    var iacc1i = Avx2.MultiplyAddAdjacent(iacc1, scales1);

                    var iaccSb = Avx2.Add(iacc0i, iacc1i);

                    var q8sSb = Avx2.Shuffle(q8s.AsInt32(), 0).AsInt16();
                    var iaccMinSb = Avx2.MultiplyAddAdjacent(q8sSb, mins01);
                    q8s = Avx2.ShiftRightLogical128BitLane(q8s.AsByte(), 4).AsInt16();

                    iaccB = Avx2.Add(iaccB, iaccSb);
                    iaccMinB = Avx2.Add(iaccMinB, iaccMinSb);
                }

                accRow = Fma.MultiplyAdd(
                    Avx.ConvertToVector256Single(iaccB), Avx.Multiply(colScale, rowScale), accRow);
                accMin = Fma.MultiplyAdd(
                    Avx.ConvertToVector256Single(iaccMinB), Avx.Multiply(colDmin, rowScale), accMin);
            }

            accRow = Avx2.PermuteVar8x32(accRow, finalpermute);
            Avx.Subtract(accRow, accMin).Store(output + x * 8);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Hi(Vector256<byte> v) => Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Sh(Vector256<byte> v, [ConstantExpected] byte imm) => Avx2.Shuffle(v.AsInt32(), imm).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<sbyte> Sh32(Vector256<sbyte> v, [ConstantExpected] byte imm) => Avx2.Shuffle(v.AsInt32(), imm).AsSByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Blend(Vector256<byte> a, Vector256<byte> b) =>
            Avx2.Blend(a.AsInt32(), b.AsInt32(), 170).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> Mul(Vector256<byte> rhs, Vector256<sbyte> lhs) =>
            Avx2.MultiplyAddAdjacent(rhs, lhs);

        private static Vector256<sbyte> BroadcastLo(sbyte* p)
        {
            var lo = Vector128.Load(p);
            return Vector256.Create(lo, lo);
        }

        private static void Unpack(uint* u, uint k1, uint k2, uint k3)
        {
            u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
            var aux = u[1] & k1;
            u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
            u[2] = aux;
            u[0] &= k1;
        }

        private static Vector256<float> LoadF16x8(byte* p)
        {
            var u = (ushort*)p;
            return Vector256.Create(
                (float)BitConverter.UInt16BitsToHalf(u[0]), (float)BitConverter.UInt16BitsToHalf(u[1]),
                (float)BitConverter.UInt16BitsToHalf(u[2]), (float)BitConverter.UInt16BitsToHalf(u[3]),
                (float)BitConverter.UInt16BitsToHalf(u[4]), (float)BitConverter.UInt16BitsToHalf(u[5]),
                (float)BitConverter.UInt16BitsToHalf(u[6]), (float)BitConverter.UInt16BitsToHalf(u[7]));
        }

        private static Vector256<float> LoadF16x8Rearrange(byte* p, Vector128<byte> deltamask)
        {
            var bytes = Ssse3.Shuffle(Vector128.Load(p), deltamask);
            var tmp = stackalloc byte[16];
            bytes.Store(tmp);
            return LoadF16x8(tmp);
        }
    }
}
