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
    /// <see cref="BoundsCheckBenchmark"/> with the payload made as cheap as possible, so the measurement can
    /// actually resolve what it claims to.
    ///
    /// <para><b>Why a second class.</b> The float version accumulates with <c>sum += data[i]</c>, and
    /// floating-point addition is a serial dependency the CPU cannot overlap — every iteration waits several
    /// cycles for the previous one. A bounds check is a compare and a perfectly-predicted branch that a
    /// wide out-of-order core issues in that shadow for free, so the float benchmark would report "no
    /// difference" whether or not the check was there. That is the scaffolding-outweighs-the-subject trap this
    /// repository has been caught by before, and reporting its 1.00 ratios as proof would be repeating it.</para>
    ///
    /// <para>Integer addition has single-cycle latency, so the dependency chain is roughly four times
    /// shorter and the loop is far closer to being bound by the index arithmetic itself. RyuJIT has no loop
    /// auto-vectoriser, so no arm gets turned into SIMD behind our backs and the comparison stays honest.</para>
    ///
    /// <para><b>RESULT (2026-07-24, Ryzen 9 9950X3D) — and read the mechanism, not the ratios.</b> At 256
    /// elements the loop runs at 0.203 ns per element, about 0.87 cycles, i.e. already at the integer
    /// dependency chain's floor. Every arm lands on 0.99–1.02. <b>That is not evidence that a bounds check is
    /// cheap: it is evidence there was no bounds check in either arm.</b></para>
    ///
    /// <para><c>--disasm</c> settles it. <see cref="ArrayLocalLength"/> compiles to a bare pointer walk —
    /// <c>add ecx,[rax]; add rax,4; dec edx; jne</c> — with no <c>cmp</c>/<c>jae</c> anywhere in the loop. And
    /// <see cref="SumWithExternalBound"/>, the arm written specifically so the check could not be eliminated,
    /// gets <b>loop cloning</b>: RyuJIT emits <c>cmp r10d,edx / jl</c> once <i>before</i> the loop — "is the
    /// array at least as long as the count?" — and then runs the same checkless pointer walk. The
    /// per-iteration <c>cmp/jae</c> and the <c>CORINFO_HELP_RNGCHKFAIL</c> call exist only in the cold clone
    /// that runs when that guard fails.</para>
    ///
    /// <para>So the practical answer to "can bounds checks be disabled for speed" is that on a sequential
    /// scan there is nothing left to disable — the JIT hoists the check out of the loop even when the bound
    /// is an unrelated parameter. Measuring the cost of a check would need a shape where cloning cannot
    /// apply, such as indirect indexing (<c>data[index[i]]</c>), where no up-front comparison can bound the
    /// subscript. That is a different benchmark, and a different question from the one anyone optimising this
    /// codebase actually has.</para>
    ///
    /// <para>Kept as a documented negative: the numbers below are a null result, and the reason they are null
    /// is in the disassembly rather than in the timings.</para>
    /// </summary>
    [SimpleJob]
    public unsafe class BoundsCheckIntegerBenchmark
    {
        private int[] _data = [];

        /// <summary>1 KB (L1), 256 KB (L2) — DRAM sizes are omitted: at that point memory latency decides
        /// everything and the question stops being about indexing.</summary>
        [Params(256, 65_536)]
        public int Length
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _data = new int[Length];
            for (var i = 0; i < Length; i++)
            {
                _data[i] = i;
            }
        }

        /// <summary>The bound is the array's own length: the JIT can prove the index and drop the check.</summary>
        [Benchmark(Baseline = true)]
        public int ArrayLocalLength()
        {
            var data = _data;
            var sum = 0;

            for (var i = 0; i < data.Length; i++)
            {
                sum += data[i];
            }

            return sum;
        }

        /// <summary>Bound arrives as a parameter, unrelated to the array — the check must stay.</summary>
        [Benchmark]
        public int ArrayParameterLength()
        {
            return SumWithExternalBound(_data, _data.Length);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static int SumWithExternalBound(int[] data, int count)
        {
            var sum = 0;

            for (var i = 0; i < count; i++)
            {
                sum += data[i];
            }

            return sum;
        }

        [Benchmark]
        public int SpanIndexed()
        {
            var span = _data.AsSpan();
            var sum = 0;

            for (var i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }

            return sum;
        }

        [Benchmark]
        public int UnsafeAdd()
        {
            var span = _data.AsSpan();
            ref var origin = ref MemoryMarshal.GetReference(span);
            var sum = 0;

            for (var i = 0; i < span.Length; i++)
            {
                sum += Unsafe.Add(ref origin, i);
            }

            return sum;
        }

        [Benchmark]
        public int Pointer()
        {
            var data = _data;
            var sum = 0;

            fixed (int* origin = data)
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
