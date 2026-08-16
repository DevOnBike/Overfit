// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Tensors;

namespace Benchmarks
{
    /// <summary>
    /// The three ways this codebase can obtain scratch, measured against each other across the sizes that
    /// actually occur: <c>stackalloc</c>, Microsoft's <c>ArrayPool&lt;T&gt;.Shared</c> used directly, and
    /// Overfit's <c>PooledBuffer&lt;T&gt;</c> wrapper over that same pool. A plain <c>new float[n]</c> is
    /// included as the do-nothing baseline everyone starts from.
    ///
    /// <para>Two questions this answers. First, what the OVERFIT025 budget actually costs when a site is
    /// pushed off the stack. Second, what the <c>PooledBuffer</c> wrapper costs over raw <c>ArrayPool</c> —
    /// the wrapper exists so RS0030 can ban the raw pool and leave one audit point, and that is only a good
    /// trade if it is free.</para>
    ///
    /// <para>Every arm allocates, zeroes, and then <b>touches</b> the buffer. Measuring an untouched buffer
    /// measures nothing: stack memory is not committed until it is written, which is precisely where the
    /// stack's apparent advantage goes.</para>
    ///
    /// <para>Deliberately not on the shared <c>BenchmarkConfig</c> — its <c>InvocationCount=1</c> job leaves
    /// sub-microsecond work measuring timer noise.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class ScratchBufferStrategyBenchmark
    {
        /// <summary>Elements. 16 floats = 64 B … 131072 floats = 512 KB, spanning the budget, the L1/L2
        /// boundary and the large-object-heap threshold.</summary>
        [Params(16, 128, 1024, 16_384, 131_072)]
        public int Length
        {
            get; set;
        }

        [Benchmark(Baseline = true)]
        public float NewArray()
        {
            var buffer = new float[Length];

            return Touch(buffer);
        }

        [Benchmark]
        public float Stackalloc()
        {
            // The library sets [module: SkipLocalsInit], so this arrives dirty and the Clear is part of the
            // cost — exactly as it is at the real call sites.
            Span<float> buffer = stackalloc float[Length];
            buffer.Clear();

            return Touch(buffer);
        }

        [Benchmark]
        public float ArrayPoolShared()
        {
            var rented = ArrayPool<float>.Shared.Rent(Length);
            try
            {
                var buffer = rented.AsSpan(0, Length);
                buffer.Clear();

                return Touch(buffer);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        [Benchmark]
        public float PooledBuffer()
        {
            using var buffer = new PooledBuffer<float>(Length, clearMemory: true);

            return Touch(buffer.Span);
        }

        /// <summary>Writes and reads every element, so the pages are actually committed.</summary>
        private static float Touch(Span<float> buffer)
        {
            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = i;
            }

            var sum = 0f;
            for (var i = 0; i < buffer.Length; i++)
            {
                sum += buffer[i];
            }

            return sum;
        }
    }
}
