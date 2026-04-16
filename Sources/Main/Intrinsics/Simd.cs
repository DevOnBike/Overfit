// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DevOnBike.Overfit.Intrinsics
{
    internal static class Simd
    {
        public static bool IsSupported => Avx.IsSupported;
        public static bool IsAvx512Supported => Avx512F.IsSupported;

        /// <summary>
        /// Minimum number of floats to use AVX-512.
        /// Below this threshold, AVX2 is faster due to lower setup overhead.
        /// </summary>
        private const int Avx512MinSize = 512; // 2KB

        // ====================================================================
        // ADD: dst = a + b
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
        {
            if (a.Length != b.Length || a.Length != dst.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            // AVX-512 path: 16 floats per iteration (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var simd512Count = Vector512<float>.Count; // 16

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector512.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx512F.Add(va, vb);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }
            // AVX2 path: 8 floats per iteration
            else if (Avx.IsSupported)
            {
                var simd256Count = Vector256<float>.Count; // 8

                for (; i <= len - simd256Count; i += simd256Count)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector256.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx.Add(va, vb);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            // Scalar fallback
            for (; i < len; i++)
            {
                Unsafe.Add(ref dRef, i) = Unsafe.Add(ref aRef, i) + Unsafe.Add(ref bRef, i);
            }
        }

        // ====================================================================
        // MULADD: dst += a * scalar (fused multiply-add)
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MulAdd(ReadOnlySpan<float> a, float scalar, Span<float> dst)
        {
            if (a.Length != dst.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            // AVX-512 path with FMA (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var simd512Count = Vector512<float>.Count; // 16
                var vs = Vector512.Create(scalar);

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vd = Vector512.LoadUnsafe(ref dRef, (nuint)i);
                    var vr = Avx512F.FusedMultiplyAdd(va, vs, vd);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }
            // AVX2 path with FMA
            else if (Avx.IsSupported)
            {
                var simd256Count = Vector256<float>.Count; // 8
                var vs = Vector256.Create(scalar);

                for (; i <= len - simd256Count; i += simd256Count)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vd = Vector256.LoadUnsafe(ref dRef, (nuint)i);

                    var vr = Fma.IsSupported
                        ? Fma.MultiplyAdd(va, vs, vd)
                        : Avx.Add(Avx.Multiply(va, vs), vd);

                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            // Scalar fallback
            for (; i < len; i++)
            {
                Unsafe.Add(ref dRef, i) += Unsafe.Add(ref aRef, i) * scalar;
            }
        }

        // ====================================================================
        // DOT: return sum(a * b)
        // ====================================================================

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

            // AVX-512 path (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var acc512 = Vector512<float>.Zero;
                var simd512Count = Vector512<float>.Count; // 16

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector512.LoadUnsafe(ref bRef, (nuint)i);
                    acc512 = Avx512F.FusedMultiplyAdd(va, vb, acc512);
                }

                // Reduce 512 -> 256 -> 128 -> scalar
                var lower256 = acc512.GetLower();
                var upper256 = acc512.GetUpper();
                var sum256 = Avx.Add(lower256, upper256);

                var lower128 = sum256.GetLower();
                var upper128 = sum256.GetUpper();
                var sum128 = Sse.Add(lower128, upper128);

                if (Sse3.IsSupported)
                {
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                }
                else
                {
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b00_01_10_11));
                }

                var sum = sum128.GetElement(0);

                // Scalar tail
                for (; i < len; i++)
                {
                    sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
                }

                return sum;
            }

            // AVX2 path
            if (Avx.IsSupported)
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

                var sum =
                    acc.GetElement(0) + acc.GetElement(1) +
                    acc.GetElement(2) + acc.GetElement(3) +
                    acc.GetElement(4) + acc.GetElement(5) +
                    acc.GetElement(6) + acc.GetElement(7);

                for (; i < len; i++)
                {
                    sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
                }

                return sum;
            }

            // Scalar fallback
            var scalarSum = 0f;
            for (; i < len; i++)
            {
                scalarSum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
            }

            return scalarSum;
        }

        // ====================================================================
        // RELU: dst = max(input, 0)
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Relu(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = input.Length;
            var i = 0;

            ref var inRef = ref MemoryMarshal.GetReference(input);
            ref var outRef = ref MemoryMarshal.GetReference(output);

            // AVX-512 path (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var simd512Count = Vector512<float>.Count; // 16
                var zero512 = Vector512<float>.Zero;

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var v = Vector512.LoadUnsafe(ref inRef, (nuint)i);
                    var r = Avx512F.Max(v, zero512);
                    r.StoreUnsafe(ref outRef, (nuint)i);
                }
            }
            // AVX2 path
            else if (Avx.IsSupported)
            {
                var simd256Count = Vector256<float>.Count; // 8
                var zero256 = Vector256<float>.Zero;

                for (; i <= len - simd256Count; i += simd256Count)
                {
                    var v = Vector256.LoadUnsafe(ref inRef, (nuint)i);
                    var r = Avx.Max(v, zero256);
                    r.StoreUnsafe(ref outRef, (nuint)i);
                }
            }

            // Scalar fallback
            for (; i < len; i++)
            {
                var x = Unsafe.Add(ref inRef, i);
                Unsafe.Add(ref outRef, i) = x > 0f ? x : 0f;
            }
        }

        // ====================================================================
        // SUBTRACT: dst = a - b
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Subtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
        {
            if (a.Length != b.Length || a.Length != dst.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            // AVX-512 path (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var simd512Count = Vector512<float>.Count;

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector512.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx512F.Subtract(va, vb);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }
            // AVX2 path
            else if (Avx.IsSupported)
            {
                var simd256Count = Vector256<float>.Count;

                for (; i <= len - simd256Count; i += simd256Count)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector256.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx.Subtract(va, vb);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            // Scalar fallback
            for (; i < len; i++)
            {
                Unsafe.Add(ref dRef, i) = Unsafe.Add(ref aRef, i) - Unsafe.Add(ref bRef, i);
            }
        }

        // ====================================================================
        // MULTIPLY: dst = a * b (element-wise)
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
        {
            if (a.Length != b.Length || a.Length != dst.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            // AVX-512 path (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var simd512Count = Vector512<float>.Count;

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector512.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx512F.Multiply(va, vb);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }
            // AVX2 path
            else if (Avx.IsSupported)
            {
                var simd256Count = Vector256<float>.Count;

                for (; i <= len - simd256Count; i += simd256Count)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector256.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx.Multiply(va, vb);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            // Scalar fallback
            for (; i < len; i++)
            {
                Unsafe.Add(ref dRef, i) = Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
            }
        }

        // ====================================================================
        // SCALE: dst = a * scalar
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Scale(ReadOnlySpan<float> a, float scalar, Span<float> dst)
        {
            if (a.Length != dst.Length)
            {
                throw new ArgumentException("Span lengths must match.");
            }

            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            // AVX-512 path (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var simd512Count = Vector512<float>.Count;
                var vs = Vector512.Create(scalar);

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vr = Avx512F.Multiply(va, vs);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }
            // AVX2 path
            else if (Avx.IsSupported)
            {
                var simd256Count = Vector256<float>.Count;
                var vs = Vector256.Create(scalar);

                for (; i <= len - simd256Count; i += simd256Count)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vr = Avx.Multiply(va, vs);
                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            // Scalar fallback
            for (; i < len; i++)
            {
                Unsafe.Add(ref dRef, i) = Unsafe.Add(ref aRef, i) * scalar;
            }
        }

        // ====================================================================
        // SUM: return sum(a)
        // ====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(ReadOnlySpan<float> a)
        {
            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);

            // AVX-512 path (only for large arrays)
            if (Avx512F.IsSupported && len >= Avx512MinSize)
            {
                var acc512 = Vector512<float>.Zero;
                var simd512Count = Vector512<float>.Count;

                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    acc512 = Avx512F.Add(acc512, va);
                }

                // Reduce
                var lower256 = acc512.GetLower();
                var upper256 = acc512.GetUpper();
                var sum256 = Avx.Add(lower256, upper256);

                var lower128 = sum256.GetLower();
                var upper128 = sum256.GetUpper();
                var sum128 = Sse.Add(lower128, upper128);

                if (Sse3.IsSupported)
                {
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                }
                else
                {
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b00_01_10_11));
                }

                var sum = sum128.GetElement(0);

                for (; i < len; i++)
                {
                    sum += Unsafe.Add(ref aRef, i);
                }

                return sum;
            }

            // AVX2 path
            if (Avx.IsSupported)
            {
                var acc = Vector256<float>.Zero;
                var simdCount = Vector256<float>.Count;

                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    acc = Avx.Add(acc, va);
                }

                var sum =
                    acc.GetElement(0) + acc.GetElement(1) +
                    acc.GetElement(2) + acc.GetElement(3) +
                    acc.GetElement(4) + acc.GetElement(5) +
                    acc.GetElement(6) + acc.GetElement(7);

                for (; i < len; i++)
                {
                    sum += Unsafe.Add(ref aRef, i);
                }

                return sum;
            }

            // Scalar fallback
            var scalarSum = 0f;
            for (; i < len; i++)
            {
                scalarSum += Unsafe.Add(ref aRef, i);
            }

            return scalarSum;
        }
    }
}