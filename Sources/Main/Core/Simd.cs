// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Core
{
    internal static class Simd
    {
        public static bool IsSupported => Avx.IsSupported;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
        {
            if (a.Length != b.Length || a.Length != dst.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var simdCount = Vector256<float>.Count;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            if (Avx.IsSupported)
            {
                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector256.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx.Add(va, vb);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref dRef, i) = Unsafe.Add(ref aRef, i) + Unsafe.Add(ref bRef, i);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MulAdd(ReadOnlySpan<float> a, float scalar, Span<float> dst)
        {
            if (a.Length != dst.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var simdCount = Vector256<float>.Count;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            if (Avx.IsSupported)
            {
                var vs = Vector256.Create(scalar);

                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vd = Vector256.LoadUnsafe(ref dRef, (nuint)i);

                    var vr = Fma.IsSupported
                        ? Fma.MultiplyAdd(va, vs, vd)
                        : Avx.Add(Avx.Multiply(va, vs), vd);

                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            for (; i < len; i++)
            {
                Unsafe.Add(ref dRef, i) += Unsafe.Add(ref aRef, i) * scalar;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);

            // =======================================================
            // Ścieżka 1: AVX-512 (Zen 4/5, nowoczesne Intel Xeon)
            // 16 floatów w jednej instrukcji
            // =======================================================
            if (Vector512.IsHardwareAccelerated)
            {
                var acc = Vector512<float>.Zero;
                var simdCount = Vector512<float>.Count; // 16

                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector512.LoadUnsafe(ref bRef, (nuint)i);

                    // FMA (Fused Multiply-Add) jest wbudowane w procesory z AVX-512
                    acc = Vector512.MultiplyAddEstimate(va, vb, acc);
                    // Alternatywnie dla pewności ścisłej FMA: Avx512F.FusedMultiplyAdd(va, vb, acc)
                }

                // Błyskawiczna suma horyzontalna w .NET 10
                var sum = Vector512.Sum(acc);

                // Resztówka (Tail)
                for (; i < len; i++)
                {
                    sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
                }

                return sum;
            }

            // =======================================================
            // Ścieżka 2: AVX2 / FMA
            // 8 floatów w jednej instrukcji
            // =======================================================
            if (Vector256.IsHardwareAccelerated)
            {
                var acc = Vector256<float>.Zero;
                var simdCount = Vector256<float>.Count; // 8

                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector256.LoadUnsafe(ref bRef, (nuint)i);

                    acc = Fma.IsSupported
                        ? Fma.MultiplyAdd(va, vb, acc)
                        : Avx.Add(acc, Avx.Multiply(va, vb));
                }

                // Zastąpienie ręcznego GetElement wbudowaną sumą!
                var sum = Vector256.Sum(acc);

                // Resztówka (Tail)
                for (; i < len; i++)
                {
                    sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
                }

                return sum;
            }

            // =======================================================
            // Ścieżka 3: Opcjonalnie SSE (Vector128) - 4 floaty
            // Wrzucam dla kompletności, jeśli np. odpalasz na starszym ARM/x86
            // =======================================================
            if (Vector128.IsHardwareAccelerated)
            {
                var acc = Vector128<float>.Zero;
                var simdCount = Vector128<float>.Count; // 4

                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector128.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector128.LoadUnsafe(ref bRef, (nuint)i);
                    acc = Vector128.MultiplyAddEstimate(va, vb, acc);
                }

                var sum = Vector128.Sum(acc);
                for (; i < len; i++) sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
                return sum;
            }

            // =======================================================
            // Ścieżka 4: Fallback Skalarny (np. brak wektoryzacji)
            // =======================================================
            var scalarSum = 0f;
            for (; i < len; i++)
            {
                scalarSum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
            }

            return scalarSum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Relu(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = input.Length;
            var simdCount = Vector256<float>.Count;
            var i = 0;

            ref var inRef = ref MemoryMarshal.GetReference(input);
            ref var outRef = ref MemoryMarshal.GetReference(output);

            if (Avx.IsSupported)
            {
                var zero = Vector256<float>.Zero;

                for (; i <= len - simdCount; i += simdCount)
                {
                    var v = Vector256.LoadUnsafe(ref inRef, (nuint)i);
                    var r = Avx.Max(v, zero);
                    r.StoreUnsafe(ref outRef, (nuint)i);
                }
            }

            for (; i < len; i++)
            {
                var x = Unsafe.Add(ref inRef, i);
                Unsafe.Add(ref outRef, i) = x > 0f ? x : 0f;
            }
        }
    }
}