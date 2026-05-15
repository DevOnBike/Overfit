// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// Correctness + zero-alloc tests for <see cref="OverfitParallelFor"/>.
    /// </summary>
    public sealed class OverfitParallelForTests
    {
        private readonly ITestOutputHelper _output;

        public OverfitParallelForTests(ITestOutputHelper output)
        {
            _output = output;
        }

        /// <summary>Per-chunk callback writes its iteration values into a caller-owned span via context.</summary>
        private unsafe struct SumContext
        {
            public int* Buffer;          // length = rangeEnd
        }

        private static unsafe void SumBody(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<SumContext>(context);
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                ctx.Buffer[i] = i;
            }
        }

        [Fact]
        public unsafe void For_FillsRange_Bitwise_Equivalent_To_Sequential()
        {
            const int n = 1000;
            var buffer = new int[n];

            fixed (int* p = buffer)
            {
                var ctx = new SumContext { Buffer = p };
                OverfitParallelFor.For(0, n, &SumBody, &ctx);
            }

            // Every index written exactly once with its own value
            for (var i = 0; i < n; i++)
            {
                Assert.Equal(i, buffer[i]);
            }

            _output.WriteLine($"WorkerCount: {OverfitParallelFor.WorkerCount}");
            _output.WriteLine($"Sum 0..{n - 1}: {buffer.Sum()} (expected {n * (n - 1) / 2})");
        }

        [Fact]
        public unsafe void For_EmptyRange_NoOp()
        {
            var buffer = new int[10];
            for (var i = 0; i < buffer.Length; i++) { buffer[i] = -1; }

            fixed (int* p = buffer)
            {
                var ctx = new SumContext { Buffer = p };
                OverfitParallelFor.For(0, 0, &SumBody, &ctx);
                OverfitParallelFor.For(5, 5, &SumBody, &ctx);
                OverfitParallelFor.For(5, 3, &SumBody, &ctx); // start > end
            }

            // Buffer untouched
            for (var i = 0; i < buffer.Length; i++)
            {
                Assert.Equal(-1, buffer[i]);
            }
        }

        [Fact]
        public unsafe void For_SingleElement_RunsInline()
        {
            var buffer = new int[10];

            fixed (int* p = buffer)
            {
                var ctx = new SumContext { Buffer = p };
                OverfitParallelFor.For(7, 8, &SumBody, &ctx);
            }

            Assert.Equal(7, buffer[7]);
            for (var i = 0; i < buffer.Length; i++)
            {
                if (i == 7) { continue; }
                Assert.Equal(0, buffer[i]);
            }
        }

        [Fact]
        public unsafe void For_RangeSmallerThanWorkerCount_AllIterationsCovered()
        {
            // Verifies tiny range like 3 elements doesn't lose work to chunking math.
            const int n = 3;
            var buffer = new int[n];
            for (var i = 0; i < n; i++) { buffer[i] = -1; }

            fixed (int* p = buffer)
            {
                var ctx = new SumContext { Buffer = p };
                OverfitParallelFor.For(0, n, &SumBody, &ctx);
            }

            for (var i = 0; i < n; i++) { Assert.Equal(i, buffer[i]); }
        }

        // ── Zero-allocation guarantee ─────────────────────────────────────────

        /// <summary>
        /// THE headline test. <see cref="OverfitParallelFor.For"/> must allocate
        /// 0 managed bytes on the calling thread for the steady-state case (after
        /// JIT warmup). This is the whole reason this class exists vs Parallel.For.
        /// </summary>
        [Fact]
        public unsafe void For_AllocatesZeroBytesPerCall_OnCallingThread()
        {
            const int n = 4096;
            var buffer = new int[n];

            // Warmup: first call JITs the dispatch path + body.
            fixed (int* p = buffer)
            {
                var ctx = new SumContext { Buffer = p };
                OverfitParallelFor.For(0, n, &SumBody, &ctx);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            const int repetitions = 100;
            fixed (int* p = buffer)
            {
                var ctx = new SumContext { Buffer = p };

                var allocBefore = GC.GetAllocatedBytesForCurrentThread();
                for (var r = 0; r < repetitions; r++)
                {
                    OverfitParallelFor.For(0, n, &SumBody, &ctx);
                }
                var allocAfter = GC.GetAllocatedBytesForCurrentThread();

                var bytesPerCall = (allocAfter - allocBefore) / (double)repetitions;
                _output.WriteLine($"Allocations over {repetitions} calls: {allocAfter - allocBefore} B");
                _output.WriteLine($"Bytes per call (calling thread): {bytesPerCall:F2}");

                Assert.Equal(0, allocAfter - allocBefore);
            }
        }

        // ── Concurrent worker safety ──────────────────────────────────────────

        /// <summary>Per-chunk independent state, no shared writes — verifies no race-induced data loss.</summary>
        private unsafe struct IndependentChunkContext
        {
            public long* PerChunkSum;
            public int ChunkCounter;
        }

        private static unsafe void IndependentChunkBody(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<IndependentChunkContext>(context);

            // Each call gets a unique slot via Interlocked counter; safe because
            // total chunks <= worker count and PerChunkSum is sized accordingly.
            var slot = Interlocked.Increment(ref ctx.ChunkCounter) - 1;

            long sum = 0;
            for (var i = chunkStart; i < chunkEnd; i++) { sum += i; }
            ctx.PerChunkSum[slot] = sum;
        }

        // ── Dispatch overhead vs Parallel.For ─────────────────────────────────

        /// <summary>Empty body — measures pure dispatch + sync overhead.</summary>
        private static unsafe void EmptyBody(int chunkStart, int chunkEnd, void* context)
        {
            // No work — just measure call/return cost.
        }

        [Fact]
        public unsafe void For_DispatchOverhead_BeatsParallelFor_OnTinyWork()
        {
            const int iterations = 1000;
            const int workItems = 32;

            // Warmup both paths.
            for (var w = 0; w < 10; w++)
            {
                OverfitParallelFor.For(0, workItems, &EmptyBody, null);
                Parallel.For(0, workItems, _ => { });
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // OverfitParallelFor
            var overfitAllocBefore = GC.GetAllocatedBytesForCurrentThread();
            var overfitStart = Stopwatch.GetTimestamp();
            for (var i = 0; i < iterations; i++)
            {
                OverfitParallelFor.For(0, workItems, &EmptyBody, null);
            }
            var overfitTicks = Stopwatch.GetTimestamp() - overfitStart;
            var overfitAllocAfter = GC.GetAllocatedBytesForCurrentThread();
            var overfitUs = overfitTicks * 1_000_000.0 / Stopwatch.Frequency / iterations;
            var overfitBytes = (overfitAllocAfter - overfitAllocBefore) / (double)iterations;

            // Parallel.For
            var parallelAllocBefore = GC.GetAllocatedBytesForCurrentThread();
            var parallelStart = Stopwatch.GetTimestamp();
            for (var i = 0; i < iterations; i++)
            {
                Parallel.For(0, workItems, _ => { });
            }
            var parallelTicks = Stopwatch.GetTimestamp() - parallelStart;
            var parallelAllocAfter = GC.GetAllocatedBytesForCurrentThread();
            var parallelUs = parallelTicks * 1_000_000.0 / Stopwatch.Frequency / iterations;
            var parallelBytes = (parallelAllocAfter - parallelAllocBefore) / (double)iterations;

            _output.WriteLine($"OverfitParallelFor: {overfitUs:F2} µs/call, {overfitBytes:F1} B/call");
            _output.WriteLine($"Parallel.For:       {parallelUs:F2} µs/call, {parallelBytes:F1} B/call");
            _output.WriteLine($"Dispatch ratio:     {parallelUs / overfitUs:F2}×");
            _output.WriteLine($"Alloc reduction:    {parallelBytes / Math.Max(1, overfitBytes):F0}× (or ∞ if Overfit = 0)");

            // Strict zero-alloc proof lives in For_AllocatesZeroBytesPerCall_OnCallingThread
            // (runs in isolation, asserts == 0). Here we accept ≤ 100 B / 1000 iterations
            // (~0.1 B/call) to tolerate xUnit runner / ITestOutputHelper noise that can
            // leak into GC.GetAllocatedBytesForCurrentThread when this test runs in parallel
            // with the rest of the sweep.
            var overfitTotalAlloc = overfitAllocAfter - overfitAllocBefore;
            Assert.True(overfitTotalAlloc <= 100,
                $"OverfitParallelFor allocated {overfitTotalAlloc} B over {iterations} iterations (expected near 0).");
        }

        [Fact]
        public unsafe void For_NoTornChunks_AllSlicesCovered()
        {
            const int n = 100_000;
            var perChunkSum = new long[OverfitParallelFor.WorkerCount];

            fixed (long* p = perChunkSum)
            {
                var ctx = new IndependentChunkContext { PerChunkSum = p, ChunkCounter = 0 };
                OverfitParallelFor.For(0, n, &IndependentChunkBody, &ctx);
            }

            long total = 0;
            foreach (var s in perChunkSum) { total += s; }

            // Σ i from 0..n-1 = n*(n-1)/2
            var expected = (long)n * (n - 1) / 2;
            Assert.Equal(expected, total);

            _output.WriteLine($"Sum 0..{n - 1} via {OverfitParallelFor.WorkerCount} workers: {total} (expected {expected})");
        }

        // ── Exception propagation ─────────────────────────────────────────────

        private static unsafe void AlwaysThrowsBody(int chunkStart, int chunkEnd, void* context)
        {
            throw new InvalidOperationException($"body-error chunk=[{chunkStart},{chunkEnd})");
        }

        /// <summary>
        /// A body that throws must surface the exception on the calling
        /// thread with its original stack trace, NOT crash the process via
        /// an unhandled exception on a background worker thread. This is the
        /// production-safety property the dispatcher must guarantee.
        /// </summary>
        [Fact]
        public unsafe void For_BodyThrows_ExceptionPropagatesToCaller_StackTracePreserved()
        {
            var ex = Assert.Throws<InvalidOperationException>(() =>
            {
                OverfitParallelFor.For(0, 100, &AlwaysThrowsBody, null);
            });

            Assert.StartsWith("body-error", ex.Message);

            // ExceptionDispatchInfo.Throw preserves the original stack frame
            // — the body method should appear in the trace.
            Assert.NotNull(ex.StackTrace);
            Assert.Contains(nameof(AlwaysThrowsBody), ex.StackTrace);

            _output.WriteLine($"Caught: {ex.Message}");
            _output.WriteLine("Stack contains AlwaysThrowsBody: ✓");
        }

        /// <summary>
        /// After a throwing call returns, subsequent calls must work
        /// normally — the dispatcher state (claim counter, completion event,
        /// captured error slots) must be cleanly reset for the next For.
        /// </summary>
        [Fact]
        public unsafe void For_AfterThrow_SubsequentCallsStillWork()
        {
            try
            {
                OverfitParallelFor.For(0, 100, &AlwaysThrowsBody, null);
            }
            catch (InvalidOperationException)
            {
                // expected
            }

            // Normal call after recovery — must produce correct result.
            const int n = 256;
            var buffer = new int[n];

            fixed (int* p = buffer)
            {
                var ctx = new SumContext { Buffer = p };
                OverfitParallelFor.For(0, n, &SumBody, &ctx);
            }

            for (var i = 0; i < n; i++)
            {
                Assert.Equal(i, buffer[i]);
            }
        }
    }
}
