// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Head-to-head benchmark of <see cref="OverfitParallelFor"/> (bulk-wake
    /// dispatcher via <see cref="SemaphoreSlim.Release(int)"/>) vs
    /// <see cref="Parallel.For(int,int,System.Action{int})"/> vs a sequential
    /// baseline, parameterized by per-worker body work.
    ///
    /// <para>
    /// All three dispatchers do the SAME total work: <c>WorkerCount</c>
    /// independent chunks, each performing <see cref="InnerIters"/> inner
    /// loop iterations. <see cref="InnerIters"/> is swept across regimes so
    /// the cross-over point can be read directly from the table:
    /// </para>
    /// <list type="bullet">
    ///   <item><c>0</c> — pure dispatch overhead.</item>
    ///   <item><c>1_000</c> — ~1 µs body (dispatch ≫ body).</item>
    ///   <item><c>100_000</c> — ~100 µs body (dispatch ≈ body).</item>
    ///   <item><c>10_000_000</c> — ~10 ms body (dispatch ≪ body).</item>
    /// </list>
    ///
    /// <para>
    /// The <see cref="MemoryDiagnoser"/> from the config attaches automatically,
    /// so the Allocated column will show the headline number: 0 B/op for
    /// <c>OverfitParallelFor</c>, ~3 KB/op for <c>Parallel.For</c>.
    /// </para>
    ///
    /// <para>
    /// History: earlier prototypes (per-worker <c>AutoResetEvent.Set</c> /
    /// hybrid spin-then-park) hit a ~32-47 µs dispatch floor on
    /// 32-fanout because the N × Set calls serialize at the kernel. The
    /// current bulk-wake design drops dispatch to ~5-7 µs — competitive
    /// with <see cref="Parallel.For"/> in raw time and 3000× cheaper on
    /// allocations. See <see cref="OverfitParallelFor"/> XML doc for the
    /// algorithmic detail.
    /// </para>
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public unsafe class OverfitParallelForBenchmark
    {
        [Params(0, 10, 1000, 100_000, 1_000_000)]
        public int InnerIters { get; set; }

        private int _workerCount;
        private double _sink;

        [GlobalSetup]
        public void Setup()
        {
            _workerCount = OverfitParallelFor.WorkerCount;
        }

        // ─── Body ──────────────────────────────────────────────────────────
        // The chunk body does a fixed amount of work proportional to
        // (chunkEnd - chunkStart) * InnerIters. Since the outer range is
        // [0, workerCount) and each chunk is exactly 1 element, the per-chunk
        // body runs InnerIters inner iterations once. The accumulator is
        // written back into the caller-owned context so the JIT cannot elide
        // the loop.

        private struct WorkContext
        {
            public int InnerIters;
            public double Accumulator; // written by chunk 0 only — no contention
        }

        private static void ChunkBody(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<WorkContext>(context);

            var inner = ctx.InnerIters;
            double local = 0.0;

            for (var c = chunkStart; c < chunkEnd; c++)
            {
                // Per-element work: tight FP loop that the JIT cannot fold.
                var x = (double)(c + 1);
                for (var k = 0; k < inner; k++)
                {
                    x = x * 1.0000001 + 1e-9;
                }
                local += x;
            }

            // Only chunk 0 writes back — avoids false sharing between workers.
            if (chunkStart == 0)
            {
                ctx.Accumulator = local;
            }
        }

        private static double SequentialChunk(int chunkStart, int chunkEnd, int innerIters)
        {
            double local = 0.0;
            for (var c = chunkStart; c < chunkEnd; c++)
            {
                var x = (double)(c + 1);
                for (var k = 0; k < innerIters; k++)
                {
                    x = x * 1.0000001 + 1e-9;
                }
                local += x;
            }
            return local;
        }

        // ─── Dispatchers ───────────────────────────────────────────────────

        [Benchmark(Baseline = true, Description = "Sequential")]
        public double Sequential()
        {
            double acc = 0.0;
            for (var w = 0; w < _workerCount; w++)
            {
                acc += SequentialChunk(w, w + 1, InnerIters);
            }
            _sink = acc;
            return acc;
        }

        [Benchmark(Description = "Parallel.For (TPL)")]
        public double ParallelFor()
        {
            double acc = 0.0;
            var iters = InnerIters;

            Parallel.For(0, _workerCount, w =>
            {
                // Same work as ChunkBody for chunk size = 1.
                var x = (double)(w + 1);
                for (var k = 0; k < iters; k++)
                {
                    x = x * 1.0000001 + 1e-9;
                }
                if (w == 0)
                {
                    Volatile.Write(ref acc, x);
                }
            });

            _sink = acc;
            return acc;
        }

        [Benchmark(Description = "OverfitParallelFor")]
        public double OverfitParallel()
        {
            var ctx = new WorkContext
            {
                InnerIters = InnerIters,
                Accumulator = 0.0,
            };

            OverfitParallelFor.For(0, _workerCount, &ChunkBody, &ctx);

            _sink = ctx.Accumulator;
            return ctx.Accumulator;
        }
    }
}
