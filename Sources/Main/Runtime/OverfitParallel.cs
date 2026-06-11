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
    /// in a hot path. This class matches <c>Parallel.For</c> in
    /// dispatch latency (~5 µs/call on a 32-logical-core Ryzen) while
    /// keeping the 0 B/call guarantee.
    /// </para>
    ///
    /// <para>
    /// <b>Design.</b> <c>N = Environment.ProcessorCount</c> persistent
    /// background threads, spawned once at class init, all park on a single
    /// shared <see cref="SemaphoreSlim"/>. Per <c>For</c> call:
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
    /// <b>Why bulk wake beats N × Set.</b> An <c>AutoResetEvent.Set</c>
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
    ///         only one <c>For</c> may execute at a time. Concurrent
    ///         callers from different threads serialize.</item>
    ///   <item><b>Non-reentrant.</b> Calling <c>For</c> recursively
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
    public static unsafe class OverfitParallel
    {
        // ── TPL-side conveniences (merged from the former OverfitParallel class, 2026-06-11) ──
        // The custom zero-alloc pool below is for HOT paths (decode/inference kernels); the members
        // here are the coarse, allocating TPL layer used by training-grade ops and orchestrators.

        /// <summary>Global degree of parallelism for the TPL side (all logical processors).</summary>
        public static readonly int MaxDegreeOfParallelism = Environment.ProcessorCount;

        /// <summary>Shared <see cref="ParallelOptions"/> for raw <c>Parallel.For</c> orchestrators.
        /// Do not mutate from call sites.</summary>
        public static readonly ParallelOptions Options = new()
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount,
        };

        /// <summary>
        /// TPL <c>Parallel.For</c> that honours <see cref="SuppressParallelismOnCurrentThread"/>: under a
        /// data-parallel replica (which owns the outer parallelism) the body runs as a plain sequential
        /// loop instead of oversubscribing the box (measured 1.8× slower when nested). For GRAPH/TRAINING
        /// ops that may execute inside <c>DataParallelTrainer</c> workers; allocates (closure/TPL) — NOT
        /// for the zero-alloc decode hot path (use the function-pointer overload below).
        /// </summary>
        public static void For(int fromInclusive, int toExclusive, Action<int> body)
        {
            if (SuppressParallelismOnCurrentThread)
            {
                for (var i = fromInclusive; i < toExclusive; i++)
                {
                    body(i);
                }
                return;
            }

            Parallel.For(fromInclusive, toExclusive, Options, body);
        }

        private const string WorkerCountEnvVar = "OVERFIT_PARALLEL_WORKERS";
        private const string DecodeWorkersEnvVar = "OVERFIT_DECODE_WORKERS";

        private static readonly int _workerCount = ResolveWorkerCount();

        /// <summary>
        /// Worker cap for small, numerous single-token decode dispatches (the FFN
        /// projection matmuls). These ~0.2 ms matmuls are dispatch-overhead bound, not
        /// bandwidth bound: measured on a 32-core box, fanning a 12.7 MB FFN matrix across
        /// all 32 workers runs it at ~11 GB/s, while a handful of workers hits ~37 GB/s
        /// (≈3×) — and end-to-end Bielik-4.5B Q4_K_M decode goes 12.55 → 14.0 tok/s (+11%).
        /// The tok/s-vs-cap curve plateaus around 10-12 on a 32-core box (6→11.8, 8→13.4,
        /// 10→14.0, 12→14.1, 32→12.5). Defaults to <c>min(WorkerCount, 10)</c>; override
        /// with <c>OVERFIT_DECODE_WORKERS</c>. Prefill / training are unaffected (they pass
        /// the full <see cref="WorkerCount"/>).
        /// </summary>
        public static int DecodeMaxWorkers { get; set; } = ResolveDecodeMaxWorkers();

        private static int ResolveDecodeMaxWorkers()
        {
            var workers = ResolveWorkerCount();
            var raw = Environment.GetEnvironmentVariable(DecodeWorkersEnvVar);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out var requested) && requested > 0)
            {
                return Math.Min(requested, workers);
            }
            return Math.Min(workers, 10);
        }

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

        private static readonly Lock _gate = new();

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

        // ── Decode spin-pool (default ON; OVERFIT_DECODE_POOL=0 opts out) ──────────
        // A SEPARATE pool of _decodePoolSize threads (== DecodeMaxWorkers) that SPIN on
        // _decodeGen instead of parking, so the ~180 tiny FFN/attention dispatches per
        // decoded token avoid a per-op kernel semaphore wake (llama.cpp "wake once, spin
        // across the graph; idle threads park"). Sized to the cap, so EVERY pool thread
        // participates in a decode dispatch — no idle spin-burn (the mistake that sank the
        // all-32 spin barrier: there 22/32 spun for nothing and starved the workers).
        // Spawned only when the flag is on; prefill / training keep the main parking pool.
        // SpinWait backs off (hot-spin → yield → Sleep(1)), so the pool cools between
        // tokens / when decode is idle. One-claim-per-generation = race-free (a worker only
        // claims after observing a fresh _decodeGen, published after the descriptors).
        private const string DecodePoolEnvVar = "OVERFIT_DECODE_POOL";
        private static readonly bool _decodePool = ResolveDecodePool();
        private static readonly int _decodePoolSize = ResolveDecodeMaxWorkers();
        private static readonly Lock _decodeGate = new();
        private static ChunkState[] _decodeChunks = [];
        private static long _decodeGen;

        // Spin-then-park: workers hot-spin for SpinBudget iterations after the last observed
        // generation, then PARK on this semaphore (0% CPU in idle — a serving container must not
        // burn cores between requests). The dispatcher wakes parked workers after bumping the
        // generation. Protocol is missed-wake-free: worker registers (parked++) BEFORE its final
        // generation re-check; dispatcher bumps the generation BEFORE reading the parked count —
        // Interlocked/Volatile fences order the two, so a worker that saw a stale generation is
        // always visible to the dispatcher's release. Stale semaphore tokens only cause a benign
        // spurious wake (loop re-checks the generation and spins/parks again).
        private static readonly SemaphoreSlim _decodeParkSemaphore = new(0);
        private static int _decodeParkedCount;

        // ~1-2 ms of Thread.SpinWait(32) — comfortably covers the µs-scale gaps between the ~180
        // per-token dispatches (so a token in flight never parks), while a quiet server parks
        // within a couple of milliseconds of the last token.
        private const int DecodeSpinBudgetIterations = 20_000;
        private static int _decodeChunkCount;
        private static PaddedCounter _decodeNextChunk;
        private static PaddedCounter _decodeRemaining;

        private static bool ResolveDecodePool()
        {
            // Default ON — measured best-of-3: Qwen3-0.6B +28% (56.7→72.3 tok/s), Phi-3.5-3.8B +3% (13.27→13.69),
            // 0 B/token preserved, bit-identical. Bigger win on small models (dispatch overhead is a larger fraction
            // of their small matmuls). Set OVERFIT_DECODE_POOL=0/false to opt out. Idle cost is ~0: the pool
            // spins-then-parks (see _decodeParkSemaphore), so there is no idle-CPU reason to disable it.
            var raw = Environment.GetEnvironmentVariable(DecodePoolEnvVar);
            return !(raw is "0" || string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase));
        }

        static OverfitParallel()
        {
            // (from the former OverfitParallel) make sure the TPL pool can actually field
            // MaxDegreeOfParallelism workers without ramp-up throttling.
            ThreadPool.GetMinThreads(out var minWorkerThreads, out var minCompletionPortThreads);
            if (minWorkerThreads < MaxDegreeOfParallelism)
            {
                ThreadPool.SetMinThreads(MaxDegreeOfParallelism, minCompletionPortThreads);
            }

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

            if (_decodePool && _decodePoolSize >= 1)
            {
                _decodeChunks = new ChunkState[_decodePoolSize];
                for (var i = 0; i < _decodePoolSize; i++)
                {
                    var thread = new Thread(DecodeWorkerLoop)
                    {
                        IsBackground = true,
                        Name = $"OverfitDecode-{i}",
                    };
                    thread.Start();
                }
            }
        }

        /// <summary>Number of persistent worker threads (== <see cref="Environment.ProcessorCount"/>).</summary>
        public static int WorkerCount => _workerCount;

        [ThreadStatic]
        private static bool _suppressOnThisThread;

        /// <summary>
        /// When set on the calling thread, every <c>For</c> invocation on that thread runs
        /// <b>inline</b> (no dispatch, no <c>_gate</c> lock) — as if the pool had a single worker.
        /// Intended for nested parallelism: when an outer parallel loop already saturates the cores
        /// (e.g. data-parallel training runs N model replicas, one per thread), each replica's inner
        /// kernels must stay single-threaded, otherwise N replicas × <see cref="WorkerCount"/> inner
        /// threads oversubscribe the CPU and serialize on the shared pool lock. Set it for the duration
        /// of the inner work and restore the previous value in a <c>finally</c>.
        /// </summary>
        public static bool SuppressParallelismOnCurrentThread
        {
            get => _suppressOnThisThread;
            set => _suppressOnThisThread = value;
        }

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
            => For(rangeStart, rangeEnd, minItemsPerWorker, _workerCount, body, context);

        /// <summary>
        /// Grained <c>For(rangeStart, rangeEnd, minItemsPerWorker, body, context)</c> with an
        /// explicit <paramref name="maxWorkers"/> cap on the chunk (worker) count. Small,
        /// numerous dispatches (single-token decode FFN matmuls) are dispatch-overhead
        /// bound, not bandwidth bound, so fanning them across all cores is a net loss —
        /// the optimum is a handful of workers (see <see cref="DecodeMaxWorkers"/>).
        /// Prefill / training keep the full pool by passing <see cref="WorkerCount"/>.
        /// </summary>
        public static void For(
            int rangeStart,
            int rangeEnd,
            int minItemsPerWorker,
            int maxWorkers,
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
            // single worker, the caller suppressed parallelism on this thread
            // (nested under an outer parallel loop), or the work is below the
            // caller's profitability grain. Skips the lock and the worker handoff.
            if (_workerCount <= 1 || _suppressOnThisThread || totalWork < 2L * minItemsPerWorker)
            {
                body(rangeStart, rangeEnd, context);
                return;
            }

            var cap = maxWorkers < 1 ? 1 : Math.Min(maxWorkers, _workerCount);
            var chunkCount = Math.Min(cap, totalWork);
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
                    _chunks[i].Error?.Throw();
                }
            }
        }

        /// <summary>
        /// Decode-pool dispatch (single-token FFN / attention projection). When
        /// <c>OVERFIT_DECODE_POOL</c> is on, routes to the spinning decode pool so the
        /// per-token burst of dispatches skips the per-op semaphore wake; otherwise
        /// delegates to the capped park path (<see cref="DecodeMaxWorkers"/>). The pool is
        /// sized to the cap, so there are no idle spinners. Same one-claim-per-generation
        /// protocol as the main pool; serialised by <c>_decodeGate</c>.
        /// </summary>
        public static void ForDecode(
            int rangeStart,
            int rangeEnd,
            delegate*<int, int, void*, void> body,
            void* context)
        {
            if (!_decodePool)
            {
                For(rangeStart, rangeEnd, 1, DecodeMaxWorkers, body, context);
                return;
            }

            if (body == null)
            {
                throw new ArgumentNullException(nameof(body));
            }

            if (rangeEnd <= rangeStart)
            {
                return;
            }

            var totalWork = rangeEnd - rangeStart;
            if (_decodePoolSize <= 1 || _suppressOnThisThread || totalWork < 2)
            {
                body(rangeStart, rangeEnd, context);
                return;
            }

            var chunkCount = Math.Min(_decodePoolSize, totalWork);
            var perChunk = (totalWork + chunkCount - 1) / chunkCount;

            lock (_decodeGate)
            {
                _decodeNextChunk.Value = 0;
                _decodeChunkCount = chunkCount;
                Volatile.Write(ref _decodeRemaining.Value, chunkCount);

                for (var i = 0; i < chunkCount; i++)
                {
                    var chunkStart = rangeStart + i * perChunk;
                    var chunkEnd = (int)Math.Min((long)chunkStart + perChunk, rangeEnd);

                    _decodeChunks[i].Start = chunkStart;
                    _decodeChunks[i].End = chunkEnd;
                    _decodeChunks[i].Body = body;
                    _decodeChunks[i].Context = context;
                    _decodeChunks[i].Error = null;
                }

                // Publish descriptors, then release the spinning pool (release fence).
                Volatile.Write(ref _decodeGen, _decodeGen + 1);

                // Wake any parked workers (idle pool). During an active decode the workers stay
                // inside their spin budget, the count is 0 and this is a single volatile read.
                var parked = Volatile.Read(ref _decodeParkedCount);
                if (parked > 0)
                {
                    _decodeParkSemaphore.Release(parked);
                }

                // Calling thread participates — greedy drain (safe under _decodeGate).
                while (true)
                {
                    var index = Interlocked.Increment(ref _decodeNextChunk.Value) - 1;
                    if (index >= _decodeChunkCount)
                    {
                        break;
                    }

                    ExecuteDecodeChunk(index);
                }

                // Pure spin — the workers are hot, so completion lands in microseconds;
                // a SpinWait that yields/sleeps would add latency to every dispatch.
                while (Volatile.Read(ref _decodeRemaining.Value) != 0)
                {
                    Thread.SpinWait(32);
                }

                for (var i = 0; i < chunkCount; i++)
                {
                    _decodeChunks[i].Error?.Throw();
                }
            }
        }

        private static void ExecuteDecodeChunk(int index)
        {
            try
            {
                _decodeChunks[index].Body(
                    _decodeChunks[index].Start, _decodeChunks[index].End, _decodeChunks[index].Context);
            }
            catch (Exception ex)
            {
                _decodeChunks[index].Error = ExceptionDispatchInfo.Capture(ex);
            }
            finally
            {
                Interlocked.Decrement(ref _decodeRemaining.Value);
            }
        }

        /// <summary>
        /// Decode spin-pool worker: poll <see cref="_decodeGen"/> with a
        /// <see cref="SpinWait"/> (hot-spin → yield → Sleep(1)), then claim ONE chunk per
        /// generation. Pool size == the decode cap, so a worker almost always gets a chunk —
        /// no idle spin-burn.
        /// </summary>
        private static void DecodeWorkerLoop()
        {
            var seen = 0L;
            while (true)
            {
                // PURE hot spin — no Sleep/Yield backoff. SpinWait.SpinOnce() escalates to
                // Sleep(1) within ~20 calls, which is fatal here: the gaps between the ~180
                // per-token dispatches (main-thread norms/attention) exceed that, so the pool
                // would sleep and each dispatch would pay a ~1 ms wake. The pool is sized to
                // the decode cap (<= cores), so keeping these few threads hot does not
                // oversubscribe — staying ready is the entire point. Burns CPU while a decode
                // is in flight (the documented OVERFIT_DECODE_POOL trade-off); between tokens
                // the loop still burns, so this flag is for throughput-mode / dedicated
                // inference, not idle embedding.
                long gen;
                var spins = 0;
                while ((gen = Volatile.Read(ref _decodeGen)) == seen)
                {
                    Thread.SpinWait(32);

                    if (++spins < DecodeSpinBudgetIterations)
                    {
                        continue;
                    }

                    // Budget exhausted — park until the next dispatch. Register FIRST, then
                    // re-check the generation (closes the race with a dispatcher that bumped
                    // it before seeing the registration).
                    Interlocked.Increment(ref _decodeParkedCount);
                    try
                    {
                        if (Volatile.Read(ref _decodeGen) == seen)
                        {
                            _decodeParkSemaphore.Wait();
                        }
                    }
                    finally
                    {
                        Interlocked.Decrement(ref _decodeParkedCount);
                    }

                    spins = 0;
                }

                seen = gen;

                var index = Interlocked.Increment(ref _decodeNextChunk.Value) - 1;
                if (index < _decodeChunkCount)
                {
                    ExecuteDecodeChunk(index);
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
                    Debug.Fail($"OverfitParallel: claim index {index} >= chunkCount {_chunkCount}.");
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
