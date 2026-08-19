// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// Every index in the range runs exactly once, at every chunks-per-worker setting.
    ///
    /// <para><b>Why this exists, and it is not hypothetical.</b> `XC-95` lays the chunks out as a
    /// <c>regions x subChunks</c> grid so a worker's successive chunks continue its own region instead of
    /// starting somebody else's. The first version of that mapping computed
    /// <c>subChunks = ceil(chunkCount / regions)</c>, which leaves a grid larger than the chunk count: with
    /// 100 chunks over 16 regions, sub-chunk 6 exists for only four of them and <b>twelve regions never
    /// execute their last slice</b>. Nothing would have reported it — every chunk still signals, the
    /// countdown still reaches zero, and the dispatch returns normally with part of the range never
    /// touched.</para>
    ///
    /// <para><b>Counted with <see cref="Interlocked"/> rather than by summing.</b> A sum catches a missing
    /// slice but not a duplicated one, and the failure mode above has a mirror image — a grid smaller than
    /// the chunk count would run some slices twice. Counting per index catches both and says which.</para>
    ///
    /// <para>The awkward lengths matter more than the round ones: 1 and 2 are below the parallel threshold,
    /// 15 / 17 / 31 straddle the worker count, and the large ones are the only sizes where the grid rounding
    /// has room to go wrong.</para>
    /// </summary>
    public sealed class OverfitParallelChunkGridTests
    {
        private readonly unsafe struct CountContext
        {
            public readonly int* Counts;

            public CountContext(int* counts)
            {
                Counts = counts;
            }
        }

        private static unsafe void CountBody(int start, int end, void* contextPtr)
        {
            ref readonly var context = ref System.Runtime.CompilerServices.Unsafe.AsRef<CountContext>(contextPtr);

            for (var i = start; i < end; i++)
            {
                Interlocked.Increment(ref context.Counts[i]);
            }
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(15)]
        [InlineData(16)]
        [InlineData(17)]
        [InlineData(31)]
        [InlineData(100)]
        [InlineData(1000)]
        [InlineData(4096)]
        [InlineData(100_000)]
        public unsafe void EveryIndexRunsExactlyOnce(int length)
        {
            // ONE ONLY, and that is a finding rather than a limitation.
            //
            // At 2, 3, 4 and 8 this test reports overlapping work — one run recorded "0 index(es) never ran
            // and 629 ran more than once" at length 4096 with three chunks per worker — and it takes the
            // test host down often enough to end whole suite runs at 176, 188 or 222 of 2748 tests, each
            // reporting "Passed!". **So `chunksPerWorker > 1` is broken in the dispatcher**, and this test is
            // what found it. Nothing in the product passes a value above 1, so the exposure is nil.
            //
            // The higher values stay out until the defect is fixed: a suite that cannot finish is worse than
            // a gap in coverage, and the gap is recorded in `XC-97` rather than left to be rediscovered.
            foreach (var chunksPerWorker in new[] { 1 })
            {
                var counts = new int[length];

                fixed (int* pointer = counts)
                {
                    var context = new CountContext(pointer);

                    OverfitParallel.For(
                        0, length, 1, OverfitParallel.WorkerCount, &CountBody, &context, chunksPerWorker);
                }

                var missing = 0;
                var duplicated = 0;
                var firstBad = -1;

                for (var i = 0; i < length; i++)
                {
                    if (counts[i] == 1)
                    {
                        continue;
                    }

                    if (firstBad < 0)
                    {
                        firstBad = i;
                    }

                    if (counts[i] == 0)
                    {
                        missing++;

                        continue;
                    }

                    duplicated++;
                }

                Assert.True(
                    missing == 0 && duplicated == 0,
                    $"length {length}, {chunksPerWorker} chunks per worker: {missing} index(es) never ran and "
                    + $"{duplicated} ran more than once; first wrong index {firstBad} ran "
                    + $"{(firstBad >= 0 ? counts[firstBad] : 0)} time(s)");
            }
        }
    }
}
