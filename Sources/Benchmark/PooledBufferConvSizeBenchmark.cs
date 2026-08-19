// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace Benchmarks
{
    /// <summary>
    /// What the convolution worker's packed-panel rent actually costs, at the size it actually uses, from the
    /// number of threads it actually uses.
    ///
    /// <para><b>Why the existing figure does not settle it.</b> <c>PooledBuffer</c>'s own remarks record
    /// "Rent+Return on ArrayPool.Shared ~4 ns regardless of size", measured 2026-05-29. The convolution
    /// worker rents <c>K * 32</c> floats — <b>147,456 floats, 589 KB</b>, at VGG-16's deepest layers — once
    /// per worker per dispatch, which is about <b>18.9 MB across 32 workers per convolution call</b>. Two
    /// things about that were never checked: whether a buffer this size hits the shared pool at all, and
    /// whether thirty-two threads renting the same size simultaneously still see 4 ns.</para>
    ///
    /// <para><b>What would refute the concern:</b> if the parallel rent costs about 4 ns per worker and
    /// allocates nothing, the pool is doing its job at this size and there is no cost to remove.</para>
    ///
    /// <para>589 KB is well over the 85,000-byte large-object threshold, so a pool miss would not merely be
    /// an allocation — it would be a large-object allocation, on every convolution, from every worker.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*PooledBufferConvSize*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [WarmupCount(25)]
    [IterationCount(30)]
    public unsafe class PooledBufferConvSizeBenchmark
    {
        /// <summary>K * Nr512 at VGG-16's deepest convolutions: 4608 * 32 floats = 589 KB.</summary>
        public const int PanelFloats = 4608 * 32;

        /// <summary>Rents chained inside one measured call, so the pool's per-core stacks reach steady state.</summary>
        public const int Rents = 64;

        private static int _workers;
        private static long _sink;

        [GlobalSetup]
        public void Setup()
        {
            _workers = OverfitParallel.MaxDegreeOfParallelism;
        }

        /// <summary>One thread, the size the conv worker uses. The 4 ns figure should reproduce here or not.</summary>
        [Benchmark(Baseline = true)]
        public void SingleThreadRent()
        {
            long local = 0;

            for (var i = 0; i < Rents; i++)
            {
                using var buffer = new PooledBuffer<float>(PanelFloats, clearMemory: false);

                local += buffer.Span.Length;
            }

            _sink += local;
        }

        /// <summary>Dispatch only, no rent — subtract this from the arm below to isolate the rent.</summary>
        [Benchmark]
        public void ParallelNoRent()
        {
            long local = 0;

            for (var i = 0; i < Rents; i++)
            {
                OverfitParallel.For(0, _workers, 1, &NoRentWorker, &local);
            }

            _sink += local;
        }

        /// <summary>What the convolution worker does: every worker rents one panel per dispatch.</summary>
        [Benchmark]
        public void ParallelRent()
        {
            long local = 0;

            for (var i = 0; i < Rents; i++)
            {
                OverfitParallel.For(0, _workers, 1, &RentWorker, &local);
            }

            _sink += local;
        }

        private static void NoRentWorker(int start, int end, void* context)
        {
            for (var i = start; i < end; i++)
            {
                _ = i;
            }
        }

        private static void RentWorker(int start, int end, void* context)
        {
            using var buffer = new PooledBuffer<float>(PanelFloats, clearMemory: false);

            // Touch the first element so the rent cannot be optimised away.
            buffer.Span[0] = start;

            for (var i = start; i < end; i++)
            {
                _ = i;
            }
        }
    }
}
