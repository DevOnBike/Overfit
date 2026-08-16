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
        // Threshold for AVX-512: only use for arrays >= 512 floats (2KB)
        // Below this, AVX2 overhead is lower
        private const int Avx512Threshold = 512;

        // On Avx512Threshold above: it is UNMEASURED and its comment is wrong. Vector512WidthBenchmark
        // measured 512-bit against 256-bit at 128 floats — a quarter of this threshold, where the comment
        // claims AVX2 wins — and 512-bit was faster in every operation: Add 0.74, MulAdd 0.80, Dot 0.84.
        // The threshold cuts off the wins it was written to protect. Left in place here rather than changed
        // in passing: Add and MulAdd are on paths this session did not census, and moving a threshold on
        // three microbenchmark points without knowing which lengths the callers pass is exactly the mistake
        // that the Dot accumulator change turned out to be.

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

            // AVX-512 path for large arrays
            if (CpuFeatures.HasAvx512 && len >= Avx512Threshold)
            {
                var simd512 = Vector512<float>.Count;
                for (; i <= len - simd512; i += simd512)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector512.LoadUnsafe(ref bRef, (nuint)i);
                    Avx512F.Add(va, vb).StoreUnsafe(ref dRef, (nuint)i);
                }
                // Fall through to scalar for remainder
            }

            // AVX2 path (original code, unchanged). CpuFeatures.HasXxx are static readonly bools the JIT
            // constant-folds, so restating the AVX-512 guard here costs nothing at runtime.
            if (!(CpuFeatures.HasAvx512 && len >= Avx512Threshold) && CpuFeatures.HasAvx)
            {
                var simdCount = Vector256<float>.Count;
                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector256.LoadUnsafe(ref bRef, (nuint)i);
                    var vr = Avx.Add(va, vb);

                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            // Scalar remainder
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
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var dRef = ref MemoryMarshal.GetReference(dst);

            // AVX-512 path for large arrays
            if (CpuFeatures.HasAvx512 && len >= Avx512Threshold)
            {
                var simd512 = Vector512<float>.Count;
                var vs = Vector512.Create(scalar);
                for (; i <= len - simd512; i += simd512)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vd = Vector512.LoadUnsafe(ref dRef, (nuint)i);

                    Avx512F.FusedMultiplyAdd(va, vs, vd).StoreUnsafe(ref dRef, (nuint)i);
                }
                // Fall through to scalar for remainder
            }

            // AVX2 path (original code, unchanged). CpuFeatures.HasXxx are static readonly bools the JIT
            // constant-folds, so restating the AVX-512 guard here costs nothing at runtime.
            if (!(CpuFeatures.HasAvx512 && len >= Avx512Threshold) && CpuFeatures.HasAvx)
            {
                var simdCount = Vector256<float>.Count;
                var vs = Vector256.Create(scalar);

                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vd = Vector256.LoadUnsafe(ref dRef, (nuint)i);
                    var vr = CpuFeatures.HasFma ? Fma.MultiplyAdd(va, vs, vd) : Avx.Add(Avx.Multiply(va, vs), vd);

                    vr.StoreUnsafe(ref dRef, (nuint)i);
                }
            }

            // Scalar remainder
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

            // AVX-512 path
            if (CpuFeatures.HasAvx512)
            {
                var acc512 = Vector512<float>.Zero;
                var simd512Count = Vector512<float>.Count;

                // ONE accumulator, deliberately — a second one was written, measured and REVERTED.
                //
                // The mechanism is real and was not in doubt: with one accumulator every FMA depends on the
                // previous one, so this loop is bound by FMA latency rather than throughput. A second
                // independent chain measured 2906 -> 1626 ns at 65536 floats and 60.5 -> 38.8 ns at 2048
                // (SimdDotAccumulatorBenchmark, Ryzen 9 9950X3D, canary stable across four runs). It also
                // showed that the win is the second CHAIN and not the wider register: two 256-bit
                // accumulators land within 1% of one 512-bit accumulator.
                //
                // It was reverted because a path census answered the question the kernel benchmark could
                // not. Dot is not on any forward/inference path at all — its only callers are backward
                // (MatMulAdd_A_BT, the im2col weight-gradient GEMM, RNN backward, depthwise-conv kernel
                // gradient). And in backward, 100% of the work is below the length where two accumulators
                // start paying: a GPT-1 training step enters Dot 9.5 million times at exactly four lengths
                // (32, 68, 128, 512) and an MNIST conv step at one (784).
                //
                // Worse than useless, then: the guarding `if (len >= 1024)` cost 9-11% at lengths 68 and 128
                // even when NOT taken — the unreached dual loop still perturbs register allocation and code
                // layout in a kernel whose whole body is ~4 ns — which works out to about +6% of the Dot
                // time in a GPT-1 step in exchange for nothing. Worth revisiting only for models with
                // dModel/dFF >= 2048, and note the Q4_K LLM training path does not come through here at all.
                for (; i <= len - simd512Count; i += simd512Count)
                {
                    var va = Vector512.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector512.LoadUnsafe(ref bRef, (nuint)i);

                    acc512 = Avx512F.FusedMultiplyAdd(va, vb, acc512);
                }

                var lower256 = acc512.GetLower();
                var upper256 = acc512.GetUpper();
                var sum256 = Avx.Add(lower256, upper256);

                var lower128 = sum256.GetLower();
                var upper128 = sum256.GetUpper();
                var sum128 = Sse.Add(lower128, upper128);

                if (CpuFeatures.HasSse3)
                {
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                }

                if (!CpuFeatures.HasSse3)
                {
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b00_01_10_11));
                }

                var sum = sum128.GetElement(0);

                for (; i < len; i++)
                {
                    sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
                }

                return sum;
            }

            // AVX2 path (original)
            if (CpuFeatures.HasAvx)
            {
                var acc = Vector256<float>.Zero;
                var simdCount = Vector256<float>.Count;

                for (; i <= len - simdCount; i += simdCount)
                {
                    var va = Vector256.LoadUnsafe(ref aRef, (nuint)i);
                    var vb = Vector256.LoadUnsafe(ref bRef, (nuint)i);

                    acc = CpuFeatures.HasFma ? Fma.MultiplyAdd(va, vb, acc) : Avx.Add(acc, Avx.Multiply(va, vb));
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

            // AVX-512 path for large arrays
            if (CpuFeatures.HasAvx512 && len >= Avx512Threshold)
            {
                var simd512 = Vector512<float>.Count;
                var zero512 = Vector512<float>.Zero;
                for (; i <= len - simd512; i += simd512)
                {
                    var v = Vector512.LoadUnsafe(ref inRef, (nuint)i);
                    Avx512F.Max(v, zero512).StoreUnsafe(ref outRef, (nuint)i);
                }
                // Fall through to scalar for remainder
            }

            // AVX2 path (original code, unchanged). CpuFeatures.HasXxx are static readonly bools the JIT
            // constant-folds, so restating the AVX-512 guard here costs nothing at runtime.
            if (!(CpuFeatures.HasAvx512 && len >= Avx512Threshold) && CpuFeatures.HasAvx)
            {
                var simdCount = Vector256<float>.Count;
                var zero = Vector256<float>.Zero;

                for (; i <= len - simdCount; i += simdCount)
                {
                    var v = Vector256.LoadUnsafe(ref inRef, (nuint)i);
                    var r = Avx.Max(v, zero);

                    r.StoreUnsafe(ref outRef, (nuint)i);
                }
            }

            // Scalar remainder
            for (; i < len; i++)
            {
                var x = Unsafe.Add(ref inRef, i);
                Unsafe.Add(ref outRef, i) = x > 0f ? x : 0f;
            }
        }
    }
}