// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;

namespace Benchmarks
{
    /// <summary>
    /// What array bounds checks actually cost, and how much of that the JIT already removes without being
    /// asked.
    ///
    /// <para><b>There is no switch.</b> .NET has no property, flag or attribute that disables bounds checking.
    /// <c>DOTNET_JitNoRangeChks</c> exists only in debug builds of the JIT and is absent from every shipping
    /// runtime. What exists instead is three things, measured here side by side: a loop shape the JIT can
    /// prove safe (so it emits no check), a loop shape it cannot (so it does), and the two ways to remove the
    /// check by giving up memory safety.</para>
    ///
    /// <para><b>The arms.</b> <see cref="ArrayLocalLength"/> is the canonical eliminable shape — the loop
    /// bound is the array's own <c>Length</c>, read through a local, which is exactly the premise the range
    /// check would test. <see cref="ArrayParameterLength"/> is the same loop with the bound arriving as a
    /// parameter: nothing in the method proves it is within the array, so the check has to stay. The gap
    /// between those two is the price of a bounds check in this codebase's units. <see cref="SpanIndexed"/>,
    /// <see cref="UnsafeAdd"/> and <see cref="Pointer"/> then show what the safe span path costs and what the
    /// two unsafe escapes buy over it — <see cref="Pointer"/> being what every hot kernel in
    /// <c>Sources/Main</c> already does.</para>
    ///
    /// <para><b>What this cannot prove on its own.</b> A timing difference is evidence about the emitted code,
    /// not a reading of it. To see whether the check was actually emitted, run with
    /// <c>--disasm --disasmDepth 1</c> and look for the <c>cmp</c>/<c>jae</c> pair before each load. The
    /// float sum is deliberately scalar — .NET will not reassociate floating-point addition, so no arm gets
    /// auto-vectorised and the comparison stays about indexing rather than about SIMD.</para>
    ///
    /// <para>The largest size is included to make a specific point: once the working set leaves cache, memory
    /// latency dwarfs everything the index does, and any difference between these arms disappears. A bounds
    /// check is only ever worth discussing on data that is already resident.</para>
    /// </summary>
    [SimpleJob]
    public unsafe class BoundsCheckBenchmark
    {
        private float[] _data = [];

        /// <summary>1 KB (L1), 256 KB (L2), 64 MB (DRAM).</summary>
        [Params(256, 65_536, 16_777_216)]
        public int Length
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _data = new float[Length];
            for (var i = 0; i < Length; i++)
            {
                _data[i] = i * 0.5f;
            }
        }

        /// <summary>Idiomatic safe C#: the bound is the array's own length, so the JIT can drop the check.</summary>
        [Benchmark(Baseline = true)]
        public float ArrayLocalLength()
        {
            var data = _data;
            var sum = 0f;

            for (var i = 0; i < data.Length; i++)
            {
                sum += data[i];
            }

            return sum;
        }

        /// <summary>
        /// The same work with the bound passed in. Nothing here relates <c>count</c> to <c>data.Length</c>, so
        /// the range check cannot be eliminated — this is the arm that actually pays for one.
        /// </summary>
        [Benchmark]
        public float ArrayParameterLength()
        {
            return SumWithExternalBound(_data, _data.Length);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float SumWithExternalBound(float[] data, int count)
        {
            var sum = 0f;

            for (var i = 0; i < count; i++)
            {
                sum += data[i];
            }

            return sum;
        }

        [Benchmark]
        public float SpanIndexed()
        {
            var span = _data.AsSpan();
            var sum = 0f;

            for (var i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }

            return sum;
        }

        /// <summary>No check, GC still tracking the reference — the safest of the two escapes.</summary>
        [Benchmark]
        public float UnsafeAdd()
        {
            var span = _data.AsSpan();
            ref var origin = ref MemoryMarshal.GetReference(span);
            var sum = 0f;

            for (var i = 0; i < span.Length; i++)
            {
                sum += Unsafe.Add(ref origin, i);
            }

            return sum;
        }

        /// <summary>Pinned raw pointers — what the Q4_K / Q6_K / Conv2D kernels in Sources/Main already use.</summary>
        [Benchmark]
        public float Pointer()
        {
            var data = _data;
            var sum = 0f;

            fixed (float* origin = data)
            {
                for (var i = 0; i < data.Length; i++)
                {
                    sum += origin[i];
                }
            }

            return sum;
        }
    }
}
