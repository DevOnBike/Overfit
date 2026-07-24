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
    /// The shape where an array bounds check survives, so its cost can finally be measured.
    ///
    /// <para><b>Why the direct scans could not answer this.</b> <see cref="BoundsCheckBenchmark"/> and
    /// <see cref="BoundsCheckIntegerBenchmark"/> both returned flat 1.00 ratios, and the disassembly showed
    /// why: RyuJIT clones the loop and hoists a single <c>cmp/jl</c> in front of it — "is the array at least
    /// as long as the count?" — then runs a checkless pointer walk. Both arms were checkless, so those
    /// benchmarks compared two identical loops.</para>
    ///
    /// <para><b>Indirect indexing defeats that.</b> In <c>data[index[i]]</c> the subscript is a value loaded
    /// at runtime, so no comparison placed before the loop can bound it. The JIT has to emit a
    /// <c>cmp</c>/<c>jae</c> against <c>data.Length</c> on <i>every</i> iteration — this is the only common
    /// shape in ordinary C# where that is true. The check on <c>index</c> itself is still eliminated, because
    /// the loop is bounded by <c>index.Length</c>, so each iteration pays for exactly one check and the
    /// comparison stays clean.</para>
    ///
    /// <para><b>Two access orders, because they answer different questions.</b> With
    /// <see cref="Shuffled"/> = false the index is the identity, so the memory pattern is a plain sequential
    /// scan and the only difference between the arms is the check — the isolated measurement.
    /// With <see cref="Shuffled"/> = true the index is a fixed permutation, which is what real gather code
    /// looks like: every load is a cache miss waiting to happen, and whatever the check costs is spent inside
    /// that shadow. If the check matters anywhere it matters in the first case; the second says whether it
    /// still matters where indirect indexing is actually used.</para>
    ///
    /// <para><b>RESULT (2026-07-24, Ryzen 9 9950X3D): the check is emitted, and it costs nothing.</b>
    /// Ratios are 0.99–1.02 across all four configurations — the same null the direct scans gave, but this
    /// time it means something, because two things were verified rather than assumed:</para>
    /// <list type="number">
    /// <item><b>The check is really there.</b> <c>--disasm</c> shows the hot loop as
    /// <c>mov r9d,[rcx]</c> / <c>cmp r9d,r10d</c> / <c>jae</c> / <c>add edx,[rax+r9*4+10]</c>, with the
    /// <c>jae</c> reaching <c>CORINFO_HELP_RNGCHKFAIL</c>. A per-iteration range check against
    /// <c>data.Length</c>, exactly as intended.</item>
    /// <item><b>The harness can see effects of this size.</b> Shuffling the index at 65 536 elements costs
    /// 14 422 ns against 11 860 ns — <b>+22%</b>, resolved cleanly at under 0.5% run-to-run spread. A
    /// benchmark that detects 22% and reports 0% for the bounds check is reporting an absence, not a
    /// blindness.</item>
    /// </list>
    ///
    /// <para>The mechanism is unsurprising once stated: <c>cmp</c> plus a never-taken <c>jae</c> is two µops
    /// on a core that issues six to eight per cycle, and the branch predictor is right every single time. It
    /// executes alongside the load it guards, and the loop is bound by the load-to-add dependency, not by
    /// issue width. The check rides for free.</para>
    ///
    /// <para><b>Conclusion for anyone hunting speed here: bounds-check elimination is not a lever on this
    /// hardware.</b> On a sequential scan the JIT removes the check before you can pay for it; on indirect
    /// indexing it keeps the check and you still do not pay for it. The reason the kernels in
    /// <c>Sources/Main</c> use raw pointers is that they need addresses for SIMD loads and pointer-based
    /// tiling — losing the bounds check is a side effect of that, never the motive. And it is a side effect
    /// with a price: with no check, an undersized buffer stops raising an exception and starts corrupting
    /// memory, which is the whole premise of OVERFIT028.</para>
    ///
    /// <para>Re-verify with <c>--disasm --disasmDepth 1</c> after any runtime upgrade: if the safe arm loses
    /// its per-iteration <c>cmp</c>/<c>jae</c>, this benchmark is back to comparing two identical loops and
    /// the numbers mean nothing again.</para>
    /// </summary>
    [SimpleJob]
    public unsafe class BoundsCheckIndirectBenchmark
    {
        private int[] _data = [];
        private int[] _index = [];

        /// <summary>1 KB (L1) and 256 KB (L2).</summary>
        [Params(256, 65_536)]
        public int Length
        {
            get; set;
        }

        /// <summary>False = identity index (sequential loads). True = fixed permutation (gather-like).</summary>
        [Params(false, true)]
        public bool Shuffled
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _data = new int[Length];
            _index = new int[Length];

            for (var i = 0; i < Length; i++)
            {
                _data[i] = i;
                _index[i] = i;
            }

            if (!Shuffled)
            {
                return;
            }

            // Fisher-Yates with a fixed seed: a permutation, so every element is still visited exactly once
            // and only the order — and therefore the cache behaviour — changes.
            var rng = new Random(20260724);
            for (var i = Length - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (_index[i], _index[j]) = (_index[j], _index[i]);
            }
        }

        /// <summary>Safe C#. The subscript comes from memory, so the check on <c>data</c> cannot be hoisted.</summary>
        [Benchmark(Baseline = true)]
        public int ArrayIndirect()
        {
            var data = _data;
            var index = _index;
            var sum = 0;

            for (var i = 0; i < index.Length; i++)
            {
                sum += data[index[i]];
            }

            return sum;
        }

        /// <summary>Same loads with the check removed, GC still tracking the reference.</summary>
        [Benchmark]
        public int UnsafeAddIndirect()
        {
            var index = _index;
            ref var origin = ref MemoryMarshal.GetArrayDataReference(_data);
            var sum = 0;

            for (var i = 0; i < index.Length; i++)
            {
                sum += Unsafe.Add(ref origin, index[i]);
            }

            return sum;
        }

        /// <summary>Pinned pointers — the form the Sources/Main kernels already use.</summary>
        [Benchmark]
        public int PointerIndirect()
        {
            var index = _index;
            var sum = 0;

            fixed (int* data = _data)
            {
                for (var i = 0; i < index.Length; i++)
                {
                    sum += data[index[i]];
                }
            }

            return sum;
        }
    }
}
