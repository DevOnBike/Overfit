// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Repacked 8×8 Q4_K GEMV — produces 8 output rows at once from a
    /// <see cref="Q4KRepack"/> <c>block_q4_Kx8</c> weight and a Q8_K activation, with the 8
    /// row dot-products accumulated in SIMD lanes (no per-row horizontal reduction).
    /// Faithful AVX2 port of llama.cpp <c>ggml_gemv_q4_K_8x8_q8_K</c> (arch/x86/repack.cpp).
    /// </summary>
    public static class Q4KGemvKernel
    {
        private const int BlockKx8Bytes = Q4KRepack.BlockKx8Bytes; // 1152
        private const int DstScalesOffset = 32;
        private const int DstQsOffset = 128;

        /// <summary>
        /// <paramref name="repacked"/> = full matrix in block_q4_Kx8 layout (from
        /// <see cref="Q4KRepack.RepackMatrix"/>). Activation already quantized to Q8_K
        /// (<paramref name="actQuants"/>/<paramref name="actScales"/>/<paramref name="actBsums"/>).
        /// Writes <paramref name="outputSize"/> dot-products. AVX2 only.
        /// </summary>
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

            ref var w = ref MemoryMarshal.GetReference(repacked);
            ref var aq = ref MemoryMarshal.GetReference(actQuants);
            ref var ab = ref MemoryMarshal.GetReference(actBsums);

            var m4b = Vector256.Create((byte)0x0F);
            var deltamask = Vector128.Create((byte)0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
            var scalemask = Vector128.Create((byte)0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7);
            var finalpermute = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);

            const uint kmask1 = 0x3f3f3f3f;
            const uint kmask2 = 0x0f0f0f0f;
            const uint kmask3 = 0x03030303;

            for (var x = 0; x < outputSize / 8; x++)
            {
                ref var bptr = ref Unsafe.Add(ref w, (nuint)((long)x * nb * BlockKx8Bytes));

                var accRow = Vector256<float>.Zero;
                var accMin = Vector256<float>.Zero;

                for (var b = 0; b < nb; b++)
                {
                    ref var blk = ref Unsafe.Add(ref bptr, (nuint)((long)b * BlockKx8Bytes));

                    var rowScale = Vector256.Create(actScales[b]);
                    var colScale = LoadF16x8Rearrange(ref blk, deltamask);              // d[8], rearranged
                    var colDmin = LoadF16x8(ref Unsafe.Add(ref blk, 16));               // dmin[8]

                    // bsums of this super-block: hadd pairs, broadcast across 256.
                    var q8sums = Vector256.LoadUnsafe(ref ab, (nuint)(b * 16)).AsInt16();
                    var q8sHadd = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
                    var q8s = Vector256.Create(q8sHadd, q8sHadd).AsInt16();

                    var iaccB = Vector256<int>.Zero;
                    var iaccMinB = Vector256<int>.Zero;

                    ref var qsBase = ref Unsafe.Add(ref blk, DstQsOffset);
                    ref var scBase = ref Unsafe.Add(ref blk, DstScalesOffset);

                    for (var sb = 0; sb < 4; sb++)
                    {
                        ref var qs = ref Unsafe.Add(ref qsBase, (nuint)(sb * 256));
                        var raw0123_0 = Vector256.LoadUnsafe(ref qs, 0);
                        var raw4567_0 = Vector256.LoadUnsafe(ref qs, 32);
                        var raw0123_1 = Vector256.LoadUnsafe(ref qs, 64);
                        var raw4567_1 = Vector256.LoadUnsafe(ref qs, 96);
                        var raw0123_2 = Vector256.LoadUnsafe(ref qs, 128);
                        var raw4567_2 = Vector256.LoadUnsafe(ref qs, 160);
                        var raw0123_3 = Vector256.LoadUnsafe(ref qs, 192);
                        var raw4567_3 = Vector256.LoadUnsafe(ref qs, 224);

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

                        // Unpack 6-bit scales/mins for the two sub-blocks of this sb.
                        Span<uint> u0 = stackalloc uint[4];
                        Span<uint> u1 = stackalloc uint[4];
                        ReadScales(ref Unsafe.Add(ref scBase, (nuint)(24 * sb)), u0);
                        ReadScales(ref Unsafe.Add(ref scBase, (nuint)(12 + sb * 24)), u1);
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

                        ref var aqb = ref Unsafe.Add(ref aq, (nuint)(b * 256 + sb * 64));
                        var l00 = BroadcastLo(ref aqb, 0);
                        var l01 = BroadcastLo(ref aqb, 16);
                        var l10 = BroadcastLo(ref aqb, 32);
                        var l11 = BroadcastLo(ref aqb, 48);

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
                Avx.Subtract(accRow, accMin).StoreUnsafe(ref MemoryMarshal.GetReference(output), (nuint)(x * 8));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Hi(Vector256<byte> v) => Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Sh(Vector256<byte> v, byte imm) => Avx2.Shuffle(v.AsInt32(), imm).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<sbyte> Sh32(Vector256<sbyte> v, byte imm) => Avx2.Shuffle(v.AsInt32(), imm).AsSByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> Blend(Vector256<byte> a, Vector256<byte> b) =>
            Avx2.Blend(a.AsInt32(), b.AsInt32(), 170).AsByte();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> Mul(Vector256<byte> rhs, Vector256<sbyte> lhs) =>
            Avx2.MultiplyAddAdjacent(rhs, lhs);

        private static Vector256<sbyte> BroadcastLo(ref sbyte p, int off)
        {
            var lo = Vector128.LoadUnsafe(ref Unsafe.Add(ref p, (nuint)off));
            return Vector256.Create(lo, lo);
        }

        private static void ReadScales(ref byte p, Span<uint> u)
        {
            u[0] = Unsafe.ReadUnaligned<uint>(ref p);
            u[1] = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref p, 4));
            u[2] = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref p, 8));
        }

        private static void Unpack(Span<uint> u, uint k1, uint k2, uint k3)
        {
            u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
            var aux = u[1] & k1;
            u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
            u[2] = aux;
            u[0] &= k1;
        }

        private static Vector256<float> LoadF16x8(ref byte p)
        {
            return Vector256.Create(
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref p)),
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 2))),
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 4))),
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 6))),
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 8))),
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 10))),
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 12))),
                (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(ref Unsafe.Add(ref p, 14))));
        }

        private static Vector256<float> LoadF16x8Rearrange(ref byte p, Vector128<byte> deltamask)
        {
            // Rearrange the 8 fp16 (16 bytes) by deltamask, then convert to float.
            var bytes = Ssse3.Shuffle(Vector128.LoadUnsafe(ref p), deltamask);
            Span<byte> tmp = stackalloc byte[16];
            bytes.StoreUnsafe(ref MemoryMarshal.GetReference(tmp));
            return LoadF16x8(ref MemoryMarshal.GetReference(tmp));
        }
    }
}
