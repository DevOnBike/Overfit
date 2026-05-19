// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// Zero-allocation parallel-for built on a persistent thread pool with a
    /// <b>bulk-wake</b> dispatcher.
    ///
    /// <para>
    /// <b>Why this exists.</b>
    /// <see cref="Parallel.For(int, int, System.Action{int})"/> allocates
    /// ~3 KB per call (closure object, internal <c>Task[]</c>, TPL
    /// bookkeeping). The 3 KB alloc breaks zero-allocation inference claims
    /// (e.g. GPT-2 0 B / generated token) the moment we parallelize anything
    /// in a hot path. This class matches <see cref="Parallel.For"/> in
    /// dispatch latency (~5 µs/call on a 32-logical-core Ryzen) while
    /// keeping the 0 B/call guarantee.
    /// </para>
    ///
    /// <para>
    /// <b>Design.</b> <c>N = Environment.ProcessorCount</c> persistent
    /// background threads, spawned once at class init, all park on a single
    /// shared <see cref="SemaphoreSlim"/>. Per <see cref="For"/> call:
    /// </para>
    /// <list type="number">
    ///   <item>Reset the work-claim counter to 0 and reset
    ///         <see cref="CountdownEvent"/> to <c>chunkCount</c>.</item>
    ///   <item>Fill <c>chunkCount</c> entries of the per-chunk descriptor
    ///         array <c>_chunks[]</c> with <c>(start, end, body, ctx)</c>.</item>
    ///   <item><see cref="SemaphoreSlim.Release(int)"/><c>(chunkCount - 1)</c>
    ///         — <b>one</b> bulk-wake call publishes the slot writes and
    ///         signals up to <c>chunkCount - 1</c> waiters; the kernel
    ///         scheduler resumes them roughly in parallel, NOT serially the
    ///         way <c>N × AutoResetEvent.Set</c> would.</item>
    ///   <item>Each woken worker
    ///         <see cref="Interlocked.Increment(ref int)"/>s the shared
    ///         claim counter to grab a unique chunk index, reads its
    ///         descriptor, runs the body, and signals
    ///         <see cref="CountdownEvent"/>.</item>
    ///   <item>The calling thread runs the final chunk itself (caller
    ///         participation — one fewer wakeup, the caller's core stays
    ///         hot), then <see cref="CountdownEvent.Wait()"/>s and propagates
    ///         the first captured exception (if any) preserving its stack
    ///         trace via <see cref="ExceptionDispatchInfo"/>.</item>
    /// </list>
    ///
    /// <para>
    /// <b>Why bulk wake beats N × Set.</b> An <see cref="AutoResetEvent.Set"/>
    /// call is a kernel-event signal that wakes exactly one waiter and
    /// returns synchronously — a serial loop of N of them costs ~N µs on
    /// Windows (32 µs floor for a 32-fanout dispatch). In contrast
    /// <see cref="SemaphoreSlim.Release(int)"/> bumps the count by N inside
    /// its internal lock and queues up to N pulses before returning;
    /// empirically the resulting wake of N waiters is much cheaper than
    /// issuing N individual wake signals on the .NET runtimes / platforms
    /// we have measured. Net effect: dispatch drops from 32-47 µs (per-worker
    /// Set on AutoResetEvent) to ~5 µs (bulk). The exact kernel mechanism
    /// underneath is a runtime/OS implementation detail, not a contract.
    /// </para>
    ///
    /// <para>
    /// <b>Exception handling.</b> Bodies are user code and can throw. A
    /// thrown exception is caught per chunk, captured into
    /// <see cref="ExceptionDispatchInfo"/> (preserves stack trace), and
    /// re-thrown on the caller thread after all chunks complete. Only the
    /// first captured exception is thrown — others are dropped (consistent
    /// with how PyTorch's <c>at::parallel_for</c> behaves). This is
    /// critical because an unhandled exception on a background worker
    /// thread would otherwise crash the entire process.
    /// </para>
    ///
    /// <para>
    /// <b>Usage.</b> Body is a static method with signature
    /// <c>void Body(int chunkStart, int chunkEnd, void* context)</c>. Caller
    /// wraps state as a value-type struct, passes
    /// <c>Unsafe.AsPointer(ref ctx)</c>. Body casts back via
    /// <c>ref Unsafe.AsRef&lt;TContext&gt;(context)</c>.
    /// </para>
    ///
    /// <para>
    /// <b>Scope and reentrancy — read this before using.</b> This dispatcher
    /// is an <i>inner</i> primitive intended for hot single-session kernels
    /// (LM head, attention, linear backward, etc.). It is:
    /// </para>
    /// <list type="bullet">
    ///   <item><b>Process-global.</b> One static worker pool of N threads
    ///         spawned lazily on first touch. All callers in the process
    ///         share these workers.</item>
    ///   <item><b>Single-in-flight.</b> A class-wide lock serializes calls —
    ///         only one <see cref="For"/> may execute at a time. Concurrent
    ///         callers from different threads serialize.</item>
    ///   <item><b>Non-reentrant.</b> Calling <see cref="For"/> recursively
    ///         from inside a body <i>deadlocks</i> (the body holds <c>_gate</c>
    ///         and a nested call tries to acquire it).</item>
    /// </list>
    ///
    /// <para>
    /// <b>Not for data-parallel outer loops.</b> If you spawn N outer training
    /// workers and each tries to dispatch via this class, they will serialize
    /// on <c>_gate</c> and you lose the outer parallelism. For data-parallel
    /// training, use plain <see cref="Thread"/> / <see cref="Task"/> for the
    /// outer fan-out and either keep inner kernels sequential or set
    /// <c>OVERFIT_PARALLEL_WORKERS=1</c> (see configuration below).
    /// </para>
    ///
    /// <para>
    /// <b>Configuration.</b> Worker count defaults to
    /// <see cref="Environment.ProcessorCount"/>. It can be overridden by
    /// setting the environment variable <c>OVERFIT_PARALLEL_WORKERS</c>
    /// (positive integer, capped at <see cref="Environment.ProcessorCount"/>)
    /// <i>before any code touches this class</i> — once the static
    /// constructor has run, the count is fixed. The default suits library
    /// users who want one inference session on a dedicated box; servers
    /// hosting many models, sandboxed processes, or coexisting with other
    /// parallel libraries should size it down explicitly.
    /// </para>
    ///
    /// <para>
    /// Workers are background threads — they die with the process.
    /// </para>
    /// </summary>
    public static unsafe class OverfitParallelFor
    {
        private const string WorkerCountEnvVar = "OVERFIT_PARALLEL_WORKERS";

        private static readonly int _workerCount = ResolveWorkerCount();

        private static int ResolveWorkerCount()
        {
            var procCount = Environment.ProcessorCount;
            var raw = Environment.GetEnvironmentVariable(WorkerCountEnvVar);
            if (string.IsNullOrEmpty(raw))
            {
                return procCount;
            }
            if (!int.TryParse(raw, out var requested) || requested <= 0)
            {
                // Bad env var value — fall back to default rather than throw
                // from a static ctor (would brick the whole runtime).
                return procCount;
            }
            return Math.Min(requested, procCount);
        }

        // Bulk-wake primitive — all workers park here, dispatcher Release(N).
        // Capacity caps at _workerCount: any single Release cannot exceed N
        // since chunkCount <= workerCount.
        private static readonly SemaphoreSlim _startSemaphore;

        // Completion counter. Initialized to 0 because Reset(chunkCount) is
        // called at the top of every For; the zero initial state is never
        // observed by Wait() (no chunks have been submitted yet).
        private static readonly CountdownEvent _completion;

        private static readonly object _gate = new();

        // Per-chunk descriptors filled by the dispatcher before Release.
        // Static and array-sized to _workerCount so allocations stay at class init.
        private static readonly ChunkState[] _chunks;

        // Number of valid descriptors for the current For. Defensive — workers
        // index-check before reading _chunks[].
        private static int _chunkCount;

        // Hot, contended atomic. Every worker Interlocked.Increment's this to
        // claim a chunk index. Padded to its own 128 B span so the cache line
        // can't be shared with neighboring static fields (which would force
        // those fields' readers/writers to invalidate on every claim). 128 B
        // rather than 64 B because some Intel CPUs prefetch adjacent L2 lines
        // in pairs (adjacent-line prefetcher) — we want a clear gap on both
        // sides.
        private static PaddedCounter _nextChunk;

        static OverfitParallelFor()
        {
            _startSemaphore = new SemaphoreSlim(0, _workerCount);
            _completion = new CountdownEvent(0);
            _chunks = new ChunkState[_workerCount];

            for (var i = 0; i < _workerCount; i++)
            {
                var thread = new Thread(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"OverfitParallel-{i}",
                };
                thread.Start();
            }
        }

        /// <summary>Number of persistent worker threads (== <see cref="Environment.ProcessorCount"/>).</summary>
        public static int WorkerCount => _workerCount;

        /// <summary>
        /// Executes <paramref name="body"/> over chunks of
        /// <c>[rangeStart, rangeEnd)</c> across the worker pool. Equivalent to
        /// the grained overload with <c>minItemsPerWorker = 1</c>.
        /// </summary>
        public static void For(
            int rangeStart,
            int rangeEnd,
            delegate*<int, int, void*, void> body,
            void* context)
            => For(rangeStart, rangeEnd, 1, body, context);

        /// <summary>
        /// Executes <paramref name="body"/> over chunks of
        /// <c>[rangeStart, rangeEnd)</c> across the worker pool. Blocks until
        /// every chunk completes. Allocates 0 managed bytes per call on the
        /// happy path; the exception path allocates one
        /// <see cref="ExceptionDispatchInfo"/> per failing chunk.
        ///
        /// <para>
        /// When the total work is below <c>2 × minItemsPerWorker</c> — or there
        /// is only one worker — the body runs inline on the calling thread: no
        /// dispatch, no lock (also making the call reentrancy-safe in that
        /// case). The right grain differs per kernel, so it is a per-call
        /// argument rather than a global constant.
        /// </para>
        ///
        /// <para>
        /// The calling thread participates — it runs one chunk itself rather
        /// than only waiting — so a dispatch wakes at most
        /// <c>WorkerCount − 1</c> background workers.
        /// </para>
        /// </summary>
        public static void For(
            int rangeStart,
            int rangeEnd,
            int minItemsPerWorker,
            delegate*<int, int, void*, void> body,
            void* context)
        {
            if (body == null)
            {
                throw new ArgumentNullException(nameof(body));
            }

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minItemsPerWorker);

            // Empty / inverted range. Computed in Int64 so a pathological
            // rangeEnd - rangeStart cannot overflow Int32 and silently skip work.
            if (rangeEnd <= rangeStart)
            {
                return;
            }

            var totalWorkLong = (long)rangeEnd - rangeStart;
            if (totalWorkLong > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(rangeEnd), "Range size exceeds Int32.MaxValue.");
            }

            var totalWork = (int)totalWorkLong;

            // Inline fast-path: nothing to gain from dispatch when there is a
            // single worker, or the work is below the caller's profitability
            // grain. Skips the lock and the worker handoff entirely.
            if (_workerCount <= 1 || totalWork < 2L * minItemsPerWorker)
            {
                body(rangeStart, rangeEnd, context);
                return;
            }

            var chunkCount = Math.Min(_workerCount, totalWork);
            var perChunk = (totalWork + chunkCount - 1) / chunkCount;

            lock (_gate)
            {
                _nextChunk.Value = 0;
                _chunkCount = chunkCount;
                _completion.Reset(chunkCount);

                for (var i = 0; i < chunkCount; i++)
                {
                    var chunkStart = rangeStart + i * perChunk;
                    var chunkEnd = (int)Math.Min((long)chunkStart + perChunk, rangeEnd);

                    _chunks[i].Start = chunkStart;
                    _chunks[i].End = chunkEnd;
                    _chunks[i].Body = body;
                    _chunks[i].Context = context;
                    _chunks[i].Error = null;
                }

                // Bulk wake — one syscall releases chunkCount - 1 tokens; the
                // semaphore's internal lock provides the release-fence so the
                // descriptor writes above are visible to workers. The calling
                // thread runs the final chunk itself (caller participation).
                var workerChunks = chunkCount - 1;
                if (workerChunks > 0)
                {
                    _startSemaphore.Release(workerChunks);
                }

                ExecuteChunk(chunkCount - 1);

                _completion.Wait();

                // Propagate the first captured exception with its original
                // stack trace. Additional captured exceptions are dropped —
                // an aggregate variant could be added if a use case appears.
                for (var i = 0; i < chunkCount; i++)
                {
                    if (_chunks[i].Error != null)
                    {
                        _chunks[i].Error.Throw();
                    }
                }
            }
        }

        /// <summary>
        /// Runs one chunk's body, capturing any thrown exception into the chunk
        /// descriptor and signalling completion. Shared by the background
        /// workers and the calling thread (caller participation).
        /// </summary>
        private static void ExecuteChunk(int index)
        {
            try
            {
                _chunks[index].Body(
                    _chunks[index].Start, _chunks[index].End, _chunks[index].Context);
            }
            catch (Exception ex)
            {
                // Capture preserves stack trace; re-thrown on the caller in For().
                // ExceptionDispatchInfo.Capture allocates, but only on the
                // exception path — happy path stays 0 B.
                _chunks[index].Error = ExceptionDispatchInfo.Capture(ex);
            }
            finally
            {
                _completion.Signal();
            }
        }

        private static void WorkerLoop()
        {
            while (true)
            {
                _startSemaphore.Wait();

                // Claim a unique chunk index. Interlocked.Increment is a
                // full fence — pairs with the semaphore release so the
                // descriptor reads below see the dispatcher's writes.
                var index = Interlocked.Increment(ref _nextChunk.Value) - 1;

                if (index < _chunkCount)
                {
                    ExecuteChunk(index);
                }
                else
                {
                    // UNREACHABLE under correct SemaphoreSlim semantics:
                    // Release(chunkCount - 1) yields exactly chunkCount - 1
                    // successful Waits, so worker claim indices are always in
                    // [0, chunkCount - 1); the caller runs the final chunk.
                    //
                    // We intentionally do NOT call _completion.Signal() here.
                    // Reset(chunkCount) sized the countdown to exactly chunkCount;
                    // a spurious extra Signal would drive it below zero and
                    // throw InvalidOperationException on a background thread —
                    // which would crash the process and mask the real bug.
                    // Hanging on Wait() is the lesser evil: it surfaces clearly
                    // in a hang dump rather than as background-thread corruption.
                    //
                    // Debug builds surface the invariant violation immediately.
                    Debug.Fail($"OverfitParallelFor: claim index {index} >= chunkCount {_chunkCount}.");
                }
            }
        }

        /// <summary>
        /// Per-chunk descriptor. Padded to one cache line because workers
        /// WRITE to <see cref="Error"/> on the exception path; without
        /// padding two adjacent chunks could share a line and their error
        /// writes would invalidate each other's reads across cores. On the
        /// happy path (read-only access by workers, single-threaded writes
        /// by dispatcher) padding is unnecessary, but harmless.
        ///
        /// Note: padding only guarantees no SHARED line between adjacent
        /// array elements <i>if the array start is cache-aligned</i>. .NET
        /// heap alignment is typically 8 B, so the first element may
        /// straddle a line; subsequent elements are then offset by a fixed
        /// amount. The padding still helps in the typical case.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct ChunkState
        {
            public int Start;                                    // 4
            public int End;                                      // 4
            public void* Context;                                // 8
            public delegate*<int, int, void*, void> Body;        // 8
            public ExceptionDispatchInfo? Error;                 // 8

            // 32 bytes used; pad to 64-byte cache line.
            private readonly long _pad1;
            private readonly long _pad2;
            private readonly long _pad3;
            private readonly long _pad4;
        }

        /// <summary>
        /// Cache-line-padded wrapper for the work-claim counter. 128 B span
        /// keeps the counter clear of neighboring static fields on both
        /// sides — important because every worker hammers this with
        /// <see cref="Interlocked.Increment(ref int)"/>, and without
        /// padding each increment would invalidate the cache line on all
        /// other cores reading whichever static field shares it.
        /// </summary>
        [StructLayout(LayoutKind.Explicit, Size = 128)]
        private struct PaddedCounter
        {
            [FieldOffset(64)]
            public int Value;
        }
    }
}
