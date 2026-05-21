// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Old-vs-new comparison of the <see cref="OverfitParallelFor"/> bulk-wake
    /// dispatcher: the current implementation (with the P1/P2/P5 tuning —
    /// workerCount fast-path, caller-participation, grain overload) against the
    /// frozen pre-tuning <see cref="OverfitParallelForLegacy"/>, plus a
    /// sequential baseline.
    ///
    /// <para>
    /// All parallel methods do the SAME total work: <c>WorkerCount</c>
    /// independent chunks of <see cref="InnerIters"/> inner iterations each.
    /// <see cref="InnerIters"/> sweeps the regimes so the dispatch-overhead vs
    /// body-work cross-over is read directly from the table:
    /// <c>0</c> = pure dispatch, <c>1_000_000</c> = dispatch ≪ body.
    /// </para>
    ///
    /// <para>
    /// The config (<see cref="DispatcherBenchmarkConfig"/>) deliberately does
    /// NOT pin <c>InvocationCount=1</c>: a single µs-scale dispatch call is far
    /// too small for BenchmarkDotNet to time without timer/scheduler noise.
    /// Letting BDN auto-scale the invocation count makes each measured block
    /// long enough — which is what the prior "minimum observed iteration time
    /// is very small" warnings were asking for.
    /// </para>
    /// </summary>
    [Config(typeof(DispatcherBenchmarkConfig))]
    public unsafe class OverfitParallelForBenchmark
    {
        [Params(0, 10, 1000, 100_000, 1_000_000)]
        public int InnerIters { get; set; }

        private int _workerCount;

        [GlobalSetup]
        public void Setup()
        {
            // Touch both dispatchers so their persistent thread pools spawn
            // before measurement starts. Both read OVERFIT_PARALLEL_WORKERS, so
            // they run with an identical worker count — a fair comparison.
            _workerCount = OverfitParallelFor.WorkerCount;
            _ = OverfitParallelForLegacy.WorkerCount;
        }

        // ─── Body ──────────────────────────────────────────────────────────
        // Per-chunk work proportional to (chunkEnd - chunkStart) * InnerIters.
        // The outer range is [0, workerCount) with one element per chunk, so
        // each body runs InnerIters inner iterations once.

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
                var x = (double)(c + 1);
                for (var k = 0; k < inner; k++)
                {
                    x = x * 1.0000001 + 1e-9;
                }
                local += x;
            }

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
            return acc;
        }

        [Benchmark(Description = "OverfitParallelFor (legacy, pre-P1/P2)")]
        public double OverfitParallelLegacy()
        {
            var ctx = new WorkContext { InnerIters = InnerIters, Accumulator = 0.0 };
            OverfitParallelForLegacy.For(0, _workerCount, &ChunkBody, &ctx);
            return ctx.Accumulator;
        }

        [Benchmark(Description = "OverfitParallelFor (current, P1/P2/P5)")]
        public double OverfitParallel()
        {
            var ctx = new WorkContext { InnerIters = InnerIters, Accumulator = 0.0 };
            OverfitParallelFor.For(0, _workerCount, &ChunkBody, &ctx);
            return ctx.Accumulator;
        }
    }

    /// <summary>
    /// Config for <see cref="OverfitParallelForBenchmark"/>. Mirrors the shared
    /// <c>BenchmarkConfig</c> but omits the <c>WithInvocationCount(1)</c> /
    /// <c>WithUnrollFactor(1)</c> pin so BDN auto-scales the invocation count —
    /// required to time a µs-scale dispatch call without noise.
    /// </summary>
    internal sealed class DispatcherBenchmarkConfig : ManualConfig
    {
        public DispatcherBenchmarkConfig()
        {
            AddJob(Job.Default
                .WithWarmupCount(5)
                .WithIterationCount(20));

            AddDiagnoser(MemoryDiagnoser.Default);
            AddColumn(RankColumn.Arabic);

            WithOrderer(new DefaultOrderer(SummaryOrderPolicy.FastestToSlowest));
        }
    }
}
