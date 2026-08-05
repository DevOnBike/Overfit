// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Intrinsics;

namespace Benchmarks
{
    /// <summary>
    /// Before/after for the second accumulator in <see cref="Simd.Dot"/>, on the shipped method rather than
    /// on a copy of it — and the sweep that decides where its threshold belongs.
    ///
    /// <para><b>Why the shipped method and not another copy.</b> <c>Vector512WidthBenchmark</c> established
    /// the direction on standalone loops, which is the right way to isolate a mechanism but is not evidence
    /// about this codebase: the real kernel carries an argument check, a length-mismatch throw, an
    /// <c>AggressiveInlining</c> attribute, a three-step horizontal reduction and a scalar tail, any of which
    /// can eat a win that a bare loop shows. <see cref="Shipped_Dot"/> calls the real thing.</para>
    ///
    /// <para><see cref="Baseline_SingleAccumulator"/> is a byte-for-byte copy of the kernel as it stood
    /// before the change, reduction and tail included, so the ratio column is the change and nothing
    /// else.</para>
    ///
    /// <para><b>The lengths are the point, not decoration.</b> The dual form was measured to LOSE at 512
    /// floats (16.0 ns against 14.7) and to win by 1.39x at 4096. The threshold has to sit where those two
    /// facts cross, and 1024 / 2048 are the only measurements that can place it. Shipping a constant chosen
    /// between two measured points by taste is what the neighbouring <c>Avx512Threshold</c> did.</para>
    ///
    /// <para>Real call sites span the range: <c>TensorMath.Algebra</c> passes K, the GEMM inner dimension
    /// (hundreds to thousands), while <c>TensorMath.DepthwiseConv</c> passes a kernel-row overlap of a few
    /// dozen. A regression on the short end is a regression in convolution training.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class SimdDotAccumulatorBenchmark
    {
        /// <summary>
        /// <b>These are the lengths the workloads actually pass</b>, taken from a path census rather than
        /// chosen: a GPT-1 training step at dModel=128/dFF=512 enters <see cref="Simd.Dot"/> 9.5 million
        /// times at exactly four lengths — 32 (head dim), 68 (vocab), 128 (dModel) and 512 (dFF) — and an
        /// MNIST conv step enters it 78k times at 784 (28x28). 2048 and 65536 are kept only to confirm the
        /// win still exists where the census says nothing goes.
        /// </summary>
        [Params(32, 68, 128, 512, 784, 2048, 65_536)]
        public int Length
        {
            get; set;
        }

        private float[] _a = [];
        private float[] _b = [];

        [GlobalSetup]
        public void Setup()
        {
            _a = new float[Length];
            _b = new float[Length];

            Fill(_a, 1);
            Fill(_b, 2);
        }

        /// <summary>The shipped kernel, threshold and all.</summary>
        [Benchmark]
        public float Shipped_Dot()
        {
            return Simd.Dot(_a, _b);
        }

        /// <summary>
        /// The kernel exactly as it was before the second accumulator: one accumulator, same reduction, same
        /// tail. Anything this arm does that the shipped one does not is the change under test.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float Baseline_SingleAccumulator()
        {
            var a = _a.AsSpan();
            var b = _b.AsSpan();
            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);

            var acc512 = Vector512<float>.Zero;
            var simd512Count = Vector512<float>.Count;

            for (; i <= len - simd512Count; i += simd512Count)
            {
                acc512 = Avx512F.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(ref aRef, (nuint)i),
                    Vector512.LoadUnsafe(ref bRef, (nuint)i),
                    acc512);
            }

            var sum256 = Avx.Add(acc512.GetLower(), acc512.GetUpper());
            var sum128 = Sse.Add(sum256.GetLower(), sum256.GetUpper());

            sum128 = Sse3.HorizontalAdd(sum128, sum128);
            sum128 = Sse3.HorizontalAdd(sum128, sum128);

            var sum = sum128.GetElement(0);

            for (; i < len; i++)
            {
                sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
            }

            return sum;
        }

        /// <summary>
        /// The dual-accumulator form with no threshold at all, so the crossover can be read directly rather
        /// than inferred from where the shipped method happens to switch. Below the threshold this is what
        /// the shipped kernel would do if the constant were lowered; above it, the two must agree.
        /// </summary>
        [Benchmark]
        public float DualAccumulator_NoThreshold()
        {
            var a = _a.AsSpan();
            var b = _b.AsSpan();
            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);

            var acc0 = Vector512<float>.Zero;
            var acc1 = Vector512<float>.Zero;
            var width = Vector512<float>.Count;
            var stride = width * 2;

            for (; i <= len - stride; i += stride)
            {
                acc0 = Avx512F.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(ref aRef, (nuint)i),
                    Vector512.LoadUnsafe(ref bRef, (nuint)i),
                    acc0);

                acc1 = Avx512F.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(ref aRef, (nuint)(i + width)),
                    Vector512.LoadUnsafe(ref bRef, (nuint)(i + width)),
                    acc1);
            }

            acc0 = Avx512F.Add(acc0, acc1);

            for (; i <= len - width; i += width)
            {
                acc0 = Avx512F.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(ref aRef, (nuint)i),
                    Vector512.LoadUnsafe(ref bRef, (nuint)i),
                    acc0);
            }

            var sum256 = Avx.Add(acc0.GetLower(), acc0.GetUpper());
            var sum128 = Sse.Add(sum256.GetLower(), sum256.GetUpper());

            sum128 = Sse3.HorizontalAdd(sum128, sum128);
            sum128 = Sse3.HorizontalAdd(sum128, sum128);

            var sum = sum128.GetElement(0);

            for (; i < len; i++)
            {
                sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
            }

            return sum;
        }

        /// <summary>
        /// <b>The question the census turned into the important one.</b> Identical to
        /// <see cref="Baseline_SingleAccumulator"/> except that the threshold test is present and NOT taken —
        /// which is what every short call now pays for a win it can never reach.
        ///
        /// <para>The census found that 100% of the elements both real workloads pass are below 1024, and
        /// that 65% of the 9.5 million calls in a GPT-1 training step are 32 floats long, where the whole
        /// kernel is about a nanosecond. A compare-and-branch is not obviously free at that scale, and if it
        /// is not, the accumulator change is a net LOSS end to end rather than the harmless nothing it
        /// looks like. Comparing against the shipped method cannot settle this — that comparison carries a
        /// 10-28% copy-versus-real offset — so both arms are copies differing in one <c>if</c>.</para>
        /// </summary>
        [Benchmark]
        public float SingleAccumulator_WithUntakenThresholdBranch()
        {
            var a = _a.AsSpan();
            var b = _b.AsSpan();
            var len = a.Length;
            var i = 0;

            ref var aRef = ref MemoryMarshal.GetReference(a);
            ref var bRef = ref MemoryMarshal.GetReference(b);

            var acc512 = Vector512<float>.Zero;
            var simd512Count = Vector512<float>.Count;

            if (len >= 1024)
            {
                var acc512Second = Vector512<float>.Zero;
                var pairStride = simd512Count * 2;

                for (; i <= len - pairStride; i += pairStride)
                {
                    acc512 = Avx512F.FusedMultiplyAdd(
                        Vector512.LoadUnsafe(ref aRef, (nuint)i),
                        Vector512.LoadUnsafe(ref bRef, (nuint)i),
                        acc512);

                    acc512Second = Avx512F.FusedMultiplyAdd(
                        Vector512.LoadUnsafe(ref aRef, (nuint)(i + simd512Count)),
                        Vector512.LoadUnsafe(ref bRef, (nuint)(i + simd512Count)),
                        acc512Second);
                }

                acc512 = Avx512F.Add(acc512, acc512Second);
            }

            for (; i <= len - simd512Count; i += simd512Count)
            {
                acc512 = Avx512F.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(ref aRef, (nuint)i),
                    Vector512.LoadUnsafe(ref bRef, (nuint)i),
                    acc512);
            }

            var sum256 = Avx.Add(acc512.GetLower(), acc512.GetUpper());
            var sum128 = Sse.Add(sum256.GetLower(), sum256.GetUpper());

            sum128 = Sse3.HorizontalAdd(sum128, sum128);
            sum128 = Sse3.HorizontalAdd(sum128, sum128);

            var sum = sum128.GetElement(0);

            for (; i < len; i++)
            {
                sum += Unsafe.Add(ref aRef, i) * Unsafe.Add(ref bRef, i);
            }

            return sum;
        }

        /// <summary>
        /// Touches no intrinsic, so no accumulator count can move it. If it drifts between runs the machine
        /// drifted and the ratios above are measuring that.
        /// </summary>
        [Benchmark]
        public float Canary_ScalarSum()
        {
            var sum = 0f;
            var span = _a.AsSpan();

            for (var i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }

            return sum;
        }

        private static void Fill(float[] data, uint seed)
        {
            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                data[i] = ((seed & 0x00FFFFFF) / 16777216f) * 2f - 1f;
            }
        }
    }
}
