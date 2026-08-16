// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;

namespace Benchmarks
{
    /// <summary>
    /// Does the explicit 512-bit width in <c>Intrinsics/Simd.cs</c> pay for itself on this hardware?
    ///
    /// <para><b>This is not the experiment that was planned, because a probe closed that one first.</b> The
    /// intended lever was <c>DOTNET_PreferredVectorBitWidth=512</c> — an environment variable rather than a
    /// rewrite — to see whether the width-agnostic <c>Vector&lt;T&gt;</c> in <c>Kernels/LinearKernels.cs</c>
    /// would widen to match. Measured on this box (AMD Ryzen 9 9950X3D, .NET 10):</para>
    ///
    /// <code>
    ///   unset -> Vector&lt;float&gt;.Count = 8  (256 bits)
    ///   128   -> Vector&lt;float&gt;.Count = 4  (128 bits)   &lt;- the knob IS live
    ///   256   -> Vector&lt;float&gt;.Count = 8  (256 bits)
    ///   512   -> Vector&lt;float&gt;.Count = 8  (256 bits)   &lt;- refused
    /// </code>
    ///
    /// <para>128 moves the width, so the knob is not being dropped; 512 is <i>declined</i>. The
    /// runtimeconfig property is declined identically. So <c>Vector&lt;T&gt;</c> is 256-bit here as a matter
    /// of runtime policy and no configuration reaches it — an A/B on that variable would have run two
    /// identical arms and reported noise, which is exactly what a dead flag did to the
    /// <c>OVERFIT_TILED_PREFILL</c> "measurement".</para>
    ///
    /// <para><b>What that leaves is a sharper question.</b> <c>Simd.cs</c> selects the 512-bit path on
    /// <c>CpuFeatures.HasAvx512</c>, which is <c>Avx512F.IsSupported</c> — "is the ISA present". The comment
    /// block in <c>CpuFeatures.cs</c> states the opposite rule two files away: <i>"the property is
    /// IsHardwareAccelerated, not IsSupported: a CPU can support a width while executing it at half
    /// rate"</i>. Meanwhile the runtime, asked directly, refuses to make 512 the default width on this part.
    /// So the codebase reaches for 512 on a presence check while the runtime's own considered answer for the
    /// same hardware is 256. That disagreement is measurable, and this class measures it.</para>
    ///
    /// <para>Each arm is a copy of the corresponding loop in <c>Simd.cs</c> with the width as the only
    /// difference — same loads, same FMA, same store, same remainder-free lengths. <c>Length</c> spans the
    /// <c>Avx512Threshold</c> of 512 floats in both directions, because a threshold that has never been
    /// measured is a guess with a comment on it ("Below this, AVX2 overhead is lower").</para>
    ///
    /// <para><b>The canary is load-bearing.</b> <see cref="Canary_ScalarSum"/> touches no intrinsic and must
    /// not move between the width arms; if it does, the box moved and the ratios are worthless. This box has
    /// already said no to AVX-512 twice (the decode port, reverted; the banded 512 prefill, 1.15x only on a
    /// path that ships off), so a third negative would be a pattern rather than a surprise — and a positive
    /// needs the canary before it is believed.</para>
    ///
    /// <para><see cref="SimpleJobAttribute"/> rather than the shared <c>BenchmarkConfig</c>: these are
    /// sub-microsecond kernels and <c>InvocationCount=1</c> would measure the timer.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
    [CategoriesColumn]
    public class Vector512WidthBenchmark
    {
        /// <summary>
        /// Spans <c>Simd.Avx512Threshold</c> (512 floats) from a quarter of it to well past L2, so the
        /// answer is a curve rather than one point. All are multiples of 16, so no arm pays a remainder the
        /// other does not.
        /// </summary>
        [Params(128, 512, 4096, 65_536)]
        public int Length
        {
            get; set;
        }

        private float[] _a = [];
        private float[] _b = [];
        private float[] _dst = [];

        [GlobalSetup]
        public void Setup()
        {
            _a = new float[Length];
            _b = new float[Length];
            _dst = new float[Length];

            Fill(_a, 1);
            Fill(_b, 2);
        }

        // ── Add ──────────────────────────────────────────────────────────────────

        [BenchmarkCategory("Add")]
        [Benchmark(Baseline = true)]
        public void Add_256()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var b = ref MemoryMarshal.GetReference<float>(_b);
            ref var d = ref MemoryMarshal.GetReference<float>(_dst);
            var width = Vector256<float>.Count;

            for (var i = 0; i <= _a.Length - width; i += width)
            {
                var va = Vector256.LoadUnsafe(ref a, (nuint)i);
                var vb = Vector256.LoadUnsafe(ref b, (nuint)i);

                Avx.Add(va, vb).StoreUnsafe(ref d, (nuint)i);
            }
        }

        [BenchmarkCategory("Add")]
        [Benchmark]
        public void Add_512()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var b = ref MemoryMarshal.GetReference<float>(_b);
            ref var d = ref MemoryMarshal.GetReference<float>(_dst);
            var width = Vector512<float>.Count;

            for (var i = 0; i <= _a.Length - width; i += width)
            {
                var va = Vector512.LoadUnsafe(ref a, (nuint)i);
                var vb = Vector512.LoadUnsafe(ref b, (nuint)i);

                Avx512F.Add(va, vb).StoreUnsafe(ref d, (nuint)i);
            }
        }

        // ── MulAdd (the shape the hot paths actually run) ─────────────────────────

        [BenchmarkCategory("MulAdd")]
        [Benchmark(Baseline = true)]
        public void MulAdd_256()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var d = ref MemoryMarshal.GetReference<float>(_dst);
            var width = Vector256<float>.Count;
            var scale = Vector256.Create(1.0001f);

            for (var i = 0; i <= _a.Length - width; i += width)
            {
                var va = Vector256.LoadUnsafe(ref a, (nuint)i);
                var vd = Vector256.LoadUnsafe(ref d, (nuint)i);

                Fma.MultiplyAdd(va, scale, vd).StoreUnsafe(ref d, (nuint)i);
            }
        }

        [BenchmarkCategory("MulAdd")]
        [Benchmark]
        public void MulAdd_512()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var d = ref MemoryMarshal.GetReference<float>(_dst);
            var width = Vector512<float>.Count;
            var scale = Vector512.Create(1.0001f);

            for (var i = 0; i <= _a.Length - width; i += width)
            {
                var va = Vector512.LoadUnsafe(ref a, (nuint)i);
                var vd = Vector512.LoadUnsafe(ref d, (nuint)i);

                Avx512F.FusedMultiplyAdd(va, scale, vd).StoreUnsafe(ref d, (nuint)i);
            }
        }

        // ── Dot (one accumulator, as Simd.cs writes it) ───────────────────────────

        [BenchmarkCategory("Dot")]
        [Benchmark(Baseline = true)]
        public float Dot_256()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var b = ref MemoryMarshal.GetReference<float>(_b);
            var width = Vector256<float>.Count;
            var acc = Vector256<float>.Zero;

            for (var i = 0; i <= _a.Length - width; i += width)
            {
                var va = Vector256.LoadUnsafe(ref a, (nuint)i);
                var vb = Vector256.LoadUnsafe(ref b, (nuint)i);

                acc = Fma.MultiplyAdd(va, vb, acc);
            }

            return Vector256.Sum(acc);
        }

        [BenchmarkCategory("Dot")]
        [Benchmark]
        public float Dot_512()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var b = ref MemoryMarshal.GetReference<float>(_b);
            var width = Vector512<float>.Count;
            var acc = Vector512<float>.Zero;

            for (var i = 0; i <= _a.Length - width; i += width)
            {
                var va = Vector512.LoadUnsafe(ref a, (nuint)i);
                var vb = Vector512.LoadUnsafe(ref b, (nuint)i);

                acc = Avx512F.FusedMultiplyAdd(va, vb, acc);
            }

            return Vector512.Sum(acc);
        }

        /// <summary>
        /// Two accumulators at 256-bit. <b>This arm exists to take the 2x away from the width if the width
        /// did not earn it.</b>
        ///
        /// <para><see cref="Dot_256"/> and <see cref="Dot_512"/> both carry a single accumulator, so each
        /// FMA depends on the previous one and the loop is bound by FMA latency rather than by throughput.
        /// Halving the iteration count by doubling the width then halves the dependency chain — which
        /// predicts exactly the 0.50 that was measured, and would predict it just as well on a machine where
        /// 512-bit ops ran at half rate. If two 256-bit accumulators reach the same place, the lever is
        /// instruction-level parallelism, not AVX-512, and the fix is one extra local rather than an ISA
        /// dependency. That distinction is the whole decision.</para>
        /// </summary>
        [BenchmarkCategory("Dot")]
        [Benchmark]
        public float Dot_256_TwoAccumulators()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var b = ref MemoryMarshal.GetReference<float>(_b);
            var width = Vector256<float>.Count;
            var step = width * 2;
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;

            for (var i = 0; i <= _a.Length - step; i += step)
            {
                acc0 = Fma.MultiplyAdd(
                    Vector256.LoadUnsafe(ref a, (nuint)i),
                    Vector256.LoadUnsafe(ref b, (nuint)i),
                    acc0);

                acc1 = Fma.MultiplyAdd(
                    Vector256.LoadUnsafe(ref a, (nuint)(i + width)),
                    Vector256.LoadUnsafe(ref b, (nuint)(i + width)),
                    acc1);
            }

            return Vector256.Sum(acc0 + acc1);
        }

        /// <summary>Two accumulators at 512-bit — does width still add anything once the chain is broken?</summary>
        [BenchmarkCategory("Dot")]
        [Benchmark]
        public float Dot_512_TwoAccumulators()
        {
            ref var a = ref MemoryMarshal.GetReference<float>(_a);
            ref var b = ref MemoryMarshal.GetReference<float>(_b);
            var width = Vector512<float>.Count;
            var step = width * 2;
            var acc0 = Vector512<float>.Zero;
            var acc1 = Vector512<float>.Zero;

            for (var i = 0; i <= _a.Length - step; i += step)
            {
                acc0 = Avx512F.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(ref a, (nuint)i),
                    Vector512.LoadUnsafe(ref b, (nuint)i),
                    acc0);

                acc1 = Avx512F.FusedMultiplyAdd(
                    Vector512.LoadUnsafe(ref a, (nuint)(i + width)),
                    Vector512.LoadUnsafe(ref b, (nuint)(i + width)),
                    acc1);
            }

            return Vector512.Sum(acc0 + acc1);
        }

        // ── Canary ────────────────────────────────────────────────────────────────

        /// <summary>
        /// Touches no intrinsic, so no width arm can change it. If this moves between runs of the same
        /// configuration, the machine moved — thermal state, background load, another process — and every
        /// ratio above is measuring that instead.
        /// </summary>
        [BenchmarkCategory("Canary")]
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
