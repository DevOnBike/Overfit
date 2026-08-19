// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// What one <see cref="OverfitParallel.For"/> costs when the workers are already awake, against what it
    /// costs when they have had time to park.
    ///
    /// <para><b>Why this exists.</b> A batch sweep showed convolution taking about 3x less time per image at
    /// batch 4 than at batch 1 — and a worker-count sweep then showed the effect vanishing entirely at one
    /// worker (6.729 ms against 6.774 ms per image across an eightfold batch). So the effect is in the
    /// parallel layer, and the leading candidate is that batch 1 issues one dispatch per measured call with a
    /// gap in between, while batch 4 issues four back to back. <b>If a parked pool costs much more to wake
    /// than a hot one, that is the whole explanation and the lever is the park policy, not the batch.</b></para>
    ///
    /// <para><b>What would refute it:</b> if the hot and cold per-dispatch costs are the same, parking is not
    /// what the batch sweep was measuring and this line of reasoning is wrong.</para>
    ///
    /// <para>The body is deliberately trivial — one add per item — so what is timed is the dispatch and not
    /// the work. `Chained` issues <see cref="Dispatches"/> of them back to back inside one measured call, so
    /// only its first can find the pool parked; `Single` issues one, and BenchmarkDotNet's own gap between
    /// iterations is what lets the pool cool.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*ParallelDispatchCost*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [WarmupCount(25)]
    [IterationCount(30)]
    public unsafe class ParallelDispatchCostBenchmark
    {
        /// <summary>Dispatches chained inside one measured call. Divide the mean by this for a per-dispatch cost.</summary>
        public const int Dispatches = 64;

        /// <summary>Items per dispatch: one per worker, so every worker is woken and the body is negligible.</summary>
        private static int _items;

        private static long _sink;

        [GlobalSetup]
        public void Setup()
        {
            _items = OverfitParallel.MaxDegreeOfParallelism;

            // Touch the pool once so class initialisation is not inside a measurement.
            long local = 0;
            OverfitParallel.For(0, _items, 1, &TrivialWorker, &local);
        }

        /// <summary>One dispatch per measured call, so the pool has BenchmarkDotNet's iteration gap to park in.</summary>
        [Benchmark]
        public void Single()
        {
            long local = 0;

            OverfitParallel.For(0, _items, 1, &TrivialWorker, &local);

            _sink += local;
        }

        /// <summary><see cref="Dispatches"/> dispatches back to back, so only the first can find the pool parked.</summary>
        [Benchmark]
        public void Chained()
        {
            long local = 0;

            for (var i = 0; i < Dispatches; i++)
            {
                OverfitParallel.For(0, _items, 1, &TrivialWorker, &local);
            }

            _sink += local;
        }

        private static void TrivialWorker(int start, int end, void* context)
        {
            // Nothing that could dominate the dispatch: one increment per item, on a per-call local.
            for (var i = start; i < end; i++)
            {
                _ = i;
            }
        }
    }
}
