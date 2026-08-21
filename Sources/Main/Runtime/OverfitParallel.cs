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

#pragma warning disable OVERFIT008 // this IS the sanctioned suppress-aware wrapper around raw TPL
            Parallel.For(fromInclusive, toExclusive, Options, body);
#pragma warning restore OVERFIT008
        }

        private const string WorkerCountEnvVar = OverfitEnvironment.ParallelWorkers;
        private const string DecodeWorkersEnvVar = OverfitEnvironment.DecodeWorkers;

        private static readonly int _workerCount = ResolveWorkerCount();

        /// <summary>
        /// Worker cap for small, numerous single-token decode dispatches (the FFN
        /// projection matmuls). These ~0.2 ms matmuls are dispatch-overhead bound, not
        /// bandwidth bound: measured 2026-06-11 on a 32-core box (not re-audited since),
        /// fanning a 12.7 MB FFN matrix across all 32 workers runs it at ~11 GB/s, while a
        /// handful of workers hits ~37 GB/s (≈3×). Defaults to <c>min(WorkerCount, 10)</c>;
        /// override with <c>OVERFIT_DECODE_WORKERS</c>. Prefill / training are unaffected
        /// (they pass the full <see cref="WorkerCount"/>).
        /// </summary>
        /// <remarks>
        /// What the cap is worth end-to-end, re-audited 2026-08-14 (<c>PB-12</c>): Bielik-4.5B Q4_K_M on a
        /// Ryzen 9 9950X3D (32 logical), .NET 10.0.8, Release, HEAD e21e7c3, ABAB, best-of-3.
        /// Isolated to this cap's ORIGINAL mechanism — decode pool OFF, cap 10 against 32 — it is
        /// <c>+3.8%</c>, every paired cycle positive (1.025 / 1.057 / 1.038). The <c>+11%</c>
        /// (12.55 → 14.0 tok/s) published here from 2026-06-11 is NOT supported as stated: the sign is
        /// real, the magnitude is not.
        /// <para>
        /// At HEAD's default the same knob measures <c>+87%</c>, because the cap now also sizes the decode
        /// spin pool — it is no longer one lever, and the two figures are not comparable. Correspondingly
        /// <c>OVERFIT_DECODE_WORKERS=32</c> costs <c>-47%</c> today (15.17 → 8.10 tok/s), which supersedes
        /// the ~-11% implied by the 2026-06-11 cap curve (6→11.8, 8→13.4, 10→14.0, 12→14.1, 32→12.5). That
        /// curve was measured before the pool existed and only its <c>32</c> endpoint has been re-measured;
        /// the plateau at 10-12 is the reason for the default and has NOT been re-checked point by point.
        /// Canonical copy: <c>docs/measured-baselines.md</c>.
        /// </para>
        /// </remarks>
        public static int DecodeMaxWorkers { get; set; } = ResolveDecodeMaxWorkers();

        private static int ResolveDecodeMaxWorkers()
        {
            var workers = ResolveWorkerCount();
            var raw = Environment.GetEnvironmentVariable(DecodeWorkersEnvVar);
            if (!string.IsNullOrEmpty(raw) && int.TryParse(raw, out var requested) && requested > 0)
            {
                // Explicit override is honoured as-is (benchmarks sweep past the cliff on purpose).
                return Math.Min(requested, workers);
            }

            // Android is big.LITTLE: half the "cores" are efficiency cores that only drag the dispatch. Measured
            // on-device (Snapdragon 7s Gen 2 = 4 big + 4 little, Qwen2.5-0.5B Q4_K_M, tok/s):
            //     workers=8 pool=ON 3.40 | workers=4 pool=ON 3.57 | workers=8 pool=OFF 3.49 | workers=4 pool=OFF 3.70
            // i.e. halving the workers is worth +5…+6 % and stacks with the pool being off (+8.8 % combined).
            // procCount/2 targets the big cluster on the usual big.LITTLE split (4+4, or 1+3+4). Modest, because
            // ARM decode is scalar/dequant-bound under Mono and saturates at ~4 effective cores either way — the
            // desktop cliff below simply does not exist here.
            if (OperatingSystem.IsAndroid())
            {
                return Math.Max(1, workers / 2);
            }

            // NEVER take every CPU. This pool SPINS, so with workers == available CPUs there is no core left
            // for the DISPATCHER and throughput collapses — measured on Qwen-3B Q4_K_M (best-of-3, 2026-07-05):
            //     CPUs  workers  tok/s        CPUs  workers  tok/s
            //        4        4   5.91           4        3   9.65   (+63%)
            //        8        8  11.69           8        7  18.78   (+61%)
            //       10       10  12.13          10        9  21.49   (+77%)
            //       32       32  14.73          32       31  23.02   (+36%)
            // The cliff is exactly at headroom == 0 and is independent of SMT and of which CCD the threads land
            // on (a 96 MB V-Cache CCD measured identical to a 32 MB one — the model dwarfs any L3). The old
            // `Min(workers, 10)` meant every box with <= 10 logical CPUs defaulted INTO the cliff.
            return Math.Min(Math.Max(1, workers - 1), 10);
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
        // tokens / when decode is idle.
        //
        // CLAIMS CARRY THEIR GENERATION, and this comment used to claim the opposite — that observing a
        // fresh _decodeGen before claiming made the protocol race-free. It did not. A worker can be
        // descheduled between reading the generation and taking an index; by the time it resumes the
        // dispatch it saw may have completed, released _decodeGate, and been replaced by the next one,
        // which rewrites _decodeChunks[] wholesale. The straggler then indexed into descriptors being
        // overwritten and could read Body from one dispatch and Context from another — observed
        // 2026-08-14 as a DivideByZeroException with an FFN dispatch frame beneath an attention body.
        // The quieter consequence was worse: its extra decrement of _decodeRemaining let a LATER
        // dispatch's wait finish early, returning a partially written output buffer with no exception at
        // all. The generation, the chunk count AND the next index now live in ONE word (_decodeClaim) so
        // all three are read and advanced by a single CAS. The chunk count joined them on 2026-08-14
        // (`XC-50`): while it lived in its own field it was published BEFORE the tag, so for the length of
        // that window the word said "generation G" while the bound already said "G+1" — see
        // DecodeChunkClaim for the full account.
        private const string DecodePoolEnvVar = OverfitEnvironment.DecodePool;
        private static readonly bool _decodePool = ResolveDecodePool();

        // Clamped to what a claim word's count field can carry. chunkCount is Math.Min(_decodePoolSize,
        // totalWork) and travels INSIDE the claim word, so bounding the pool size here — once, at
        // resolution — makes an over-wide count unrepresentable instead of something every dispatch has to
        // test for. ResolveDecodeMaxWorkers already caps at Environment.ProcessorCount, so on any real box
        // this clamp is inert; it exists so that silent truncation of the bound (which would revive
        // `XC-50`) cannot become the failure mode if that cap ever changes.
        private static readonly int _decodePoolSize = ClampDecodePoolSize(ResolveDecodeMaxWorkers());

        /// <summary>
        /// Clamps a resolved decode-pool size to what a claim word's count field can carry
        /// (<see cref="DecodeChunkClaim.MaxChunkCount"/>).
        ///
        /// <para><b>It is a named method rather than an inline <c>Math.Min</c> so it can be driven from a
        /// test.</b> On every real box the clamp is inert — <see cref="ResolveDecodeMaxWorkers"/> already
        /// caps at <see cref="Environment.ProcessorCount"/>, which is five orders of magnitude below the
        /// field's limit — so a test that reads <c>_decodePoolSize</c> would assert nothing about the
        /// clamp and would additionally depend on the machine it ran on. An unexercised clamp is the same
        /// defect class as an untested bound: if <see cref="ResolveDecodeMaxWorkers"/> ever grows a
        /// configuration path, silent truncation of the count field revives `XC-50`.</para>
        /// </summary>
        /// <param name="resolvedWorkers">The pool size before clamping.</param>
        /// <returns>The pool size, never above <see cref="DecodeChunkClaim.MaxChunkCount"/>.</returns>
        internal static int ClampDecodePoolSize(int resolvedWorkers)
        {
            return Math.Min(resolvedWorkers, DecodeChunkClaim.MaxChunkCount);
        }

        /// <summary>
        /// Diagnostics only: the resolved decode-pool size actually in force, after
        /// <see cref="ResolveDecodeMaxWorkers"/> and <see cref="ClampDecodePoolSize"/>.
        ///
        /// <para><b>It exists so a test can tell whether it exercised the dispatcher at all.</b> When this
        /// is <c>1</c> — a 2-vCPU box, or <c>OVERFIT_DECODE_WORKERS=1</c> — <see cref="ForDecode"/> runs the
        /// body inline and never publishes a dispatch, so every assertion a dispatcher test makes passes
        /// without the subject having run. <see cref="DecodeMaxWorkers"/> is <b>not</b> a substitute: it is
        /// a settable public property that the resolved field does not track.</para>
        /// </summary>
        internal static int DecodePoolSize => _decodePoolSize;

        /// <summary>
        /// Diagnostics only: whether the decode spin pool is on (<c>OVERFIT_DECODE_POOL=0</c> turns it off,
        /// and it is off by default on Android). When it is off, <see cref="ForDecode"/> delegates to the
        /// capped park path and the decode claim protocol never runs — the other half of the capability
        /// question <see cref="DecodePoolSize"/> answers.
        /// </summary>
        internal static bool DecodePoolEnabled => _decodePool;

        private static readonly Lock _decodeGate = new();
        private static ChunkState[] _decodeChunks = [];
        private static long _decodeGen;

        // Spin-then-park: workers hot-spin for SpinBudget iterations after the last observed generation,
        // then PARK here (0% CPU in idle — a serving container must not burn cores between requests).
        //
        // A MONITOR ON A PREDICATE, not a counted semaphore, and the difference was worth 14.85 cores.
        // The previous protocol counted parked workers and released exactly that many tokens. A worker
        // that registered itself, then observed the generation move and skipped its Wait, left its token
        // in the semaphore permanently — nothing ever consumed it except a LATER park, which returned
        // immediately. This comment used to call that "a benign spurious wake"; it is not benign, because
        // tokens accumulate faster than parks consume them, and the pool stops sleeping altogether.
        // Measured 2026-08-14 by `DecodePoolIdleBurnTests`: 44.61 s of CPU across a 3 s idle window,
        // 14.85 effective cores, on a pool that is supposed to be at zero.
        //
        // Monitor.Wait re-tests the condition it slept on, so neither a missed pulse nor a surplus one
        // can strand a worker or spin it. The dispatcher pulses unconditionally rather than consulting a
        // parked count: deciding from a count read outside the lock races with a worker that has decided
        // to park and not yet blocked, and an uncontended Monitor acquisition costs ~20 ns against a
        // dispatch that does microseconds of work.
        // Deliberately `object` and not `System.Threading.Lock`, which is this repository's default
        // elsewhere: `lock` on a `Lock` does NOT go through Monitor, so `Monitor.Wait`/`Monitor.PulseAll`
        // on the same instance would operate on a different mechanism than the critical section itself and
        // the park protocol would be silently broken. The compiler says so as CS9216 — heeded rather than
        // suppressed, because a warning about the wrong locking primitive is the warning to believe.
        private static readonly object _decodeParkLock = new();

        // ~1-2 ms of Thread.SpinWait(32) — comfortably covers the µs-scale gaps between the ~180
        // per-token dispatches (so a token in flight never parks), while a quiet server parks
        // within a couple of milliseconds of the last token.
        private const int DecodeSpinBudgetIterations = 20_000;
        private static PaddedClaim _decodeClaim;
        private static PaddedCounter _decodeRemaining;

        private static bool ResolveDecodePool()
        {
            // Default ON. Re-measured 2026-08-14 (`PB-12`) on a Ryzen 9 9950X3D (32 logical), .NET 10.0.8,
            // Release, HEAD e21e7c3 — 24 processes, ABAB at process level, best-of-3 within each, a canary
            // before and after every timed block: Qwen3-0.6B Q8_0 +25.1% (56.56→70.75 tok/s), Q4_K_M +23.0%;
            // 0 B/token preserved, bit-identical. This supersedes the `+28% (56.7→72.3)` published here from
            // 2026-06-11, which recorded no quantisation.
            //
            // Phi-3.5-mini 3.8B is NEUTRAL TO NEGATIVE, not the `+3%` published here until 2026-08-14: the
            // sign reversed, to -1.9% (13.38→13.12 tok/s). That point estimate is itself inside this box's
            // +-3-4% cross-process floor, so no number is quoted — but it is not the box moving, because the
            // untouched arm reproduced to 0.8% while the treated arm fell 4.2%. The mechanism is a WRONG
            // DENOMINATOR: on Phi-3.5 the pool captures only ~75% of dispatches, because 52 per token still
            // go through OverfitParallel.For via the GQA For(0, KvHeadCount, ...) in CachedMultiHeadAttention.
            //
            // Bigger win on small models (dispatch overhead is a larger fraction of their small matmuls).
            // Set OVERFIT_DECODE_POOL=0/false to opt out. The pool spins-then-parks (see _decodeParkLock) so
            // idle cost is meant to be ~0 — but that has NOT been demonstrated: `DecodePoolIdleBurnTests`
            // measures the whole PROCESS, so inside a 52-test run it charges other tests' threads to the pool
            // and reports 15-17 cores; run alone it is green. See `TG-T13`. Full table, dispatch census and
            // method: docs/measured-baselines.md.
            var raw = Environment.GetEnvironmentVariable(DecodePoolEnvVar);
            if (!string.IsNullOrEmpty(raw))
            {
                return !(raw is "0" || string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase));
            }

            // ...but OFF by default on Android. Measured on-device (Motorola Edge 50 Fusion, Snapdragon 7s Gen 2,
            // 4×A78 @2.4GHz + 4×A55 @1.96GHz, Qwen2.5-0.5B Q4_K_M): pool ON→OFF is +2.6…+3.6 % — and it stops the
            // pool hot-spinning the efficiency cores, which on a battery-powered device is the real argument.
            // The reason the desktop's big pool win does not transfer: under Mono on ARM there are NO SIMD
            // intrinsics (AdvSimd64=False, Dp=False → the decode path is SCALAR), so decode is dequant-bound and
            // tops out at ~4 effective cores no matter how many workers spin. Dispatch is not the bottleneck here.
            return !OperatingSystem.IsAndroid();
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

            // Max count is the widest release a fan-out can make, which is chunkCount - 1 and chunkCount
            // is now up to _workerCount * MaxChunkFactor. It was _workerCount until 2026-08-19, and raising
            // ChunkFactor above 1 without this threw SemaphoreFullException on the first dispatch — 100 tests
            // red, which is the whole reason the factor ships defaulting to 1.
            _startSemaphore = new SemaphoreSlim(0, (int)Math.Min((long)_workerCount * MaxChunkFactor, int.MaxValue));
            _completion = new CountdownEvent(0);
            // Sized for the widest split the factor allows, not for the worker count: a fan-out may now
            // create several chunks per worker so the claim counter has something to rebalance.
            _chunks = new ChunkState[(long)_workerCount * MaxChunkFactor];

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
        /// Diagnostics only: count every real fan-out (the inline fast path is not counted, since it costs
        /// nothing to launch). Off by default and checked before the interlocked increment.
        ///
        /// <para>Exists because a prefill's fixed cost had to be attributed. Prompt-length sweeps showed
        /// ~175 ms that does not scale with the prompt — 8% of a 672-token prefill but ~70% of a chat-sized
        /// one — and the two candidates were per-dispatch launch overhead and the unavoidable walk over the
        /// weight matrix. Counting the dispatches turns that from an argument into arithmetic.</para>
        /// </summary>
        public static bool CountDispatches;

        private static long _dispatchCount;

        /// <summary>Fan-outs since the last <see cref="ResetDispatchCount"/>.</summary>
        public static long DispatchCount => Interlocked.Read(ref _dispatchCount);

        /// <summary>Clears the dispatch counter.</summary>
        public static void ResetDispatchCount()
        {
            Interlocked.Exchange(ref _dispatchCount, 0);
        }

        /// <summary>
        /// Records, per fan-out, how much of the reserved worker time was actually spent executing.
        ///
        /// <para><b>Why a hardware profiler cannot answer this.</b> AMD uProf's counters key on
        /// <c>CYCLES_NOT_IN_HALT</c>, and <b>a worker parked waiting for work produces no samples at all</b>.
        /// Measured 2026-08-19 on the 60.9 MB CNN, our code's CPI is 0.4413 at one core and 0.4205 at
        /// sixteen — <b>identical</b>, so nothing about the executed instructions degrades with thread count.
        /// Yet convolution scales 7.08x where ONNX Runtime's scales 12.12x. The missing time is either halted
        /// cores or synchronisation, and the profiler can see only the second.</para>
        ///
        /// <para>Off by default and checked before any timestamp is taken, so the dispatch path is unchanged
        /// when it is off.</para>
        /// </summary>
        public static bool MeasureOccupancy =
            Environment.GetEnvironmentVariable(OverfitEnvironment.ParallelOccupancy) == "1";

        private static long _occupancyWallTicks;
        private static long _occupancyBusyTicks;
        private static long _occupancyWorstTicks;

        /// <summary>Each dispatch's worst chunk, weighted by that dispatch's own chunk count. The
        /// straggler ratio needs this; <see cref="_occupancyWorstTicks"/> stays RAW because the
        /// overhead figure is `wall - worst` and wants the critical path, not a weighted sum. They
        /// were one field until 2026-08-21, and re-purposing it drove overhead to -1887%.</summary>
        private static long _occupancyWorstWeightedTicks;
        private static long _occupancyReservedTicks;
        private static long _occupancyDispatches;
        private static long _occupancyChunks;

        /// <summary>Busy-weighted sum of each dispatch's granularity reference, and the busy time it covers.
        /// The reference is `widestChunkItems * chunkCount / totalItems` — what the straggler WOULD read on
        /// perfectly uniform per-item cost, given integer item counts. Reported next to the straggler so a
        /// structural number is not read as imbalance. It is NOT a lower bound: the straggler falls below it
        /// whenever the widest chunk is not the slowest.</summary>
        private static double _occupancyFloorWeighted;

        private static long _occupancyFloorBusy;

        /// <summary>
        /// Upper edges of the per-dispatch straggler buckets. The aggregate ratio cannot say whether an
        /// imbalance is spread across every dispatch or concentrated in a few, and those two call for
        /// opposite work: a broad one is a property of the split, a narrow one is a property of a handful of
        /// layers. Both the COUNT and the BUSY TIME per bucket are kept, because the count says how many
        /// dispatches are unbalanced and the busy time says how much of the run they are.
        /// </summary>
        private static readonly double[] OccupancyBucketEdges = [1.05, 1.15, 1.3, 1.6, 2.5];

        private static readonly long[] _occupancyBucketCounts = new long[OccupancyBucketEdges.Length + 1];

        private static readonly long[] _occupancyBucketBusy = new long[OccupancyBucketEdges.Length + 1];

        private static double _occupancyWorstDispatchRatio;

        private static int _occupancyWorstDispatchChunks;

        private static long _occupancyWorstDispatchBusy;

        private static long _occupancyWorstDispatchItems;

        private static long _occupancyWorstDispatchWidest;

        /// <summary>
        /// Adds one fan-out to the totals. Called with <see cref="_gate"/> held, which is also what makes the
        /// plain increments safe — every dispatch on the main pool is serialised by it.
        /// </summary>
        private static void RecordOccupancy(int chunkCount, long wallTicks)
        {
            long busy = 0;
            long worst = 0;
            long items = 0;
            long widest = 0;

            for (var i = 0; i < chunkCount; i++)
            {
                var ticks = _chunks[i].BusyTicks;
                busy += ticks;

                if (ticks > worst)
                {
                    worst = ticks;
                }

                // Item counts as well as times. A chunk holds `ceil(totalItems / chunkCount)` items and the
                // last one holds the remainder, so on perfectly uniform per-item cost the worst chunk is
                // `widest * chunkCount / items` times the mean. Worked example: 100 items over 32 chunks is
                // 4 * 32 / 100 = 1.28x with no imbalance at all, which is why a straggler figure alone
                // cannot be read as evidence of imbalance.
                //
                // It is a REFERENCE VALUE UNDER UNIFORMITY, not a lower bound, and calling it a floor was
                // wrong — measured 2026-08-21 on the 60.9 MB CNN at 16 workers, the straggler is 1.27x
                // against a value of 1.35x. Below it, which a bound could not be. The straggler falls below
                // whenever the WIDEST chunk is not the SLOWEST, i.e. when per-item costs vary in a way that
                // partly cancels the width imbalance.
                //
                // Read the two together. At 32 workers the same model gives 1.02x here against a straggler
                // of 1.34x: chunks are near enough one item each, so granularity explains almost none of it
                // and the imbalance is real. At 16 workers many dispatches have fewer items than workers, so
                // the widths are forced uneven and this rises to 1.35x.
                var width = _chunks[i].End - _chunks[i].Start;
                items += width;

                if (width > widest)
                {
                    widest = width;
                }
            }

            AddOccupancySample(chunkCount, wallTicks, busy, worst, items, widest);
        }

        /// <summary>
        /// Folds one dispatch's summary into the totals. Separated from <see cref="RecordOccupancy"/> so the
        /// arithmetic can be tested with exact integers and no threads.
        ///
        /// <para><b>Why that separation exists.</b> The property that matters — balanced work reports 1.00
        /// whatever mix of chunk counts the dispatches have — is arithmetic. Asserting it through real
        /// parallel execution measures the box's load instead, and that mistake was made and caught here on
        /// 2026-08-21: the end-to-end version passed alone and failed in one full-suite run out of two,
        /// which is the `TG-T12` shape this repository already has a row for.</para>
        /// </summary>
        internal static void AddOccupancySample(
            int chunkCount, long wallTicks, long busy, long worst, long items = 0, long widest = 0)
        {
            // Busy-weighted so it is directly comparable with the straggler ratio, which is also a
            // busy-weighted quantity. Zero items means the caller did not supply widths — the arithmetic
            // tests do that deliberately — and such a sample contributes nothing to the floor.
            if (items > 0 && widest > 0)
            {
                _occupancyFloorWeighted += (double)widest * chunkCount / items * busy;
                _occupancyFloorBusy += busy;
            }

            _occupancyWallTicks += wallTicks;
            _occupancyBusyTicks += busy;

            // Weighted by THIS dispatch's chunk count, not summed raw and multiplied by the global mean
            // later. For a perfectly balanced dispatch `worst == busy / chunkCount`, so `worst * chunkCount`
            // is exactly `busy` and the ratio in OccupancyReport cancels to 1.00 whatever mix of chunk
            // counts the model produces.
            //
            // Summing `worst` raw was WRONG whenever dispatches differ in chunk count, and they do: the
            // 60.9 MB CNN reports mean chunks 26.4 on a 32-worker pool, so some fan-outs create far fewer
            // chunks than others. Worked example of the old form — two dispatches, BOTH perfectly balanced,
            // equal busy time B, one with 4 chunks and one with 48: sum of worsts is B/4 + B/48 = 0.2708B,
            // the global mean is 26, and the report claimed 0.2708B * 26 / 2B = 3.52x. Balanced work,
            // reported as a threefold straggler. The small-chunk dispatches dominated the numerator because
            // their `worst` is a large fraction of their own busy time.
            _occupancyWorstTicks += worst;
            _occupancyWorstWeightedTicks += worst * chunkCount;

            // Per-dispatch ratio, bucketed. Same definition as the aggregate, applied to one fan-out:
            // 1.00 means this dispatch's chunks all finished together.
            if (busy > 0)
            {
                var ratio = (double)worst * chunkCount / busy;
                var bucket = OccupancyBucketEdges.Length;

                for (var i = 0; i < OccupancyBucketEdges.Length; i++)
                {
                    if (ratio < OccupancyBucketEdges[i])
                    {
                        bucket = i;

                        break;
                    }
                }

                _occupancyBucketCounts[bucket]++;
                _occupancyBucketBusy[bucket] += busy;

                if (ratio > _occupancyWorstDispatchRatio)
                {
                    _occupancyWorstDispatchRatio = ratio;
                    _occupancyWorstDispatchChunks = chunkCount;
                    _occupancyWorstDispatchBusy = busy;
                    _occupancyWorstDispatchItems = items;
                    _occupancyWorstDispatchWidest = widest;
                }
            }
            _occupancyReservedTicks += (long)chunkCount * wallTicks;
            _occupancyChunks += chunkCount;
            _occupancyDispatches++;
        }

        /// <summary>
        /// The chunks-per-worker a call site gets when it asks for the tuned value rather than naming one.
        /// <b>1 reproduces the behaviour every version before 2026-08-19 had</b>, and it is the default for
        /// every call site that does not opt in.
        ///
        /// <para><b>What it is for.</b> The pool already claims dynamically — <see cref="WorkerLoop"/> takes
        /// one chunk, runs it and comes back for another — but with exactly one chunk per worker there is
        /// nothing left to come back for. Measured 2026-08-19 on the 60.9 MB CNN at 16 cores: <b>pool use
        /// 65.0%</b>, with a <b>straggler ratio of 1.28x</b>, so a fifth of the reserved worker time is spent
        /// at the barrier waiting for whichever chunk drew the slow core. Finer chunks give the claim counter
        /// something to rebalance.</para>
        ///
        /// <para><b>Two ways this can cost rather than pay</b>, both to be checked in the measurement rather
        /// than argued away: every claim is an interlocked increment on one counter, so more chunks means
        /// more traffic on <see cref="_nextChunk"/>; and a caller that rents a scratch buffer per invocation
        /// — the convolution worker rents up to 590 KB — pays that rent once per chunk, not once per
        /// dispatch.</para>
        ///
        /// <para><b>It is opt-in PER CALL SITE, and that is not caution — a global switch corrupts memory.</b>
        /// Applying it to every fan-out turned 100 tests red, and the two that survived the first fix showed
        /// why: a body may derive a slot index from its chunk ordinal and size its scratch by the worker
        /// count. <c>TensorMath.LayerNorm</c>'s backward does exactly that — partial buffers of
        /// <c>WorkerCount x C</c>, dispatched over <c>numRows</c>, with
        /// <c>chunkIdx = chunkStart / perChunk</c>. More chunks than workers makes that index run past the
        /// buffer, and it is a pinned write, so the failure is a corrupted heap rather than a wrong number.
        /// One arm of the sweep took the test host down mid-run.</para>
        ///
        /// <para><b>The dispatcher cannot detect this.</b> It sees a function pointer; nothing in the
        /// signature says whether the body treats its chunk as a worker slot. So the caller declares it, and
        /// the default stays at the value every existing body was written against.</para>
        /// </summary>
        public static readonly int ChunkFactor = ResolveChunkFactor();

        /// <summary>Upper bound on <see cref="ChunkFactor"/>, which sizes <see cref="_chunks"/>.</summary>
        private const int MaxChunkFactor = 8;

        /// <summary>
        /// Set <c>OVERFIT_PARALLEL_REGION_MAJOR=1</c> to lay the chunks out as a <c>regions x subChunks</c>
        /// grid — chunk <c>i</c> is region <c>i % regions</c>, sub-chunk <c>i / regions</c> — instead of as
        /// the <c>i</c>th consecutive slice.
        ///
        /// <para><b>What it is trying to buy.</b> Claims are handed out in index order, so with plain
        /// slicing a worker's successive chunks land wherever the counter has reached, which is a different
        /// part of the range each time. Measured 2026-08-19 with uProf at eight chunks per worker: supply
        /// from this core's L2 fell 5.7% while same-CCX L3 rose 40.3%, another CCD's cache 62.1% and DRAM
        /// 70.0%, with the instruction count flat — the same work, further from the core, CPI +10.8%. The
        /// grid is meant to make chunk <c>k + regions</c> the continuation of the region that produced chunk
        /// <c>k</c>.</para>
        ///
        /// <para><b>It is a switch rather than a code swap, and that is not tidiness.</b> The first attempt
        /// swapped the loop, rebuilt, and measured — and the measurement was worthless, because a mutation
        /// harness had restored a source file with <c>shutil.copy2</c>'s preserved timestamp and MSBuild
        /// skipped the rebuild. Both layouts in one binary, with the live one echoed in
        /// <see cref="OccupancyReport"/>, is what makes the arm checkable instead of assumed.</para>
        ///
        /// <para><b>MEASURED, AND OFF BY DEFAULT ANYWAY.</b> Three ABAB passes per model, every one negative:
        /// the 60.9 MB CNN goes 18.89 to <b>18.34 ms (-2.9%)</b> with pool use 64.8% to 69.4%, and VGG-16
        /// goes 27.31 to <b>26.74 ms (-2.1%)</b> with pool use 76.9% to 79.5%. Set to 1 to measure it.</para>
        ///
        /// <para><b>Why a measured win ships off.</b> With it on and four chunks per worker, three
        /// independent cold test runs ended at 176, 188 and 222 of 2748 tests, each reporting "Passed!", and
        /// the cause was never localised. Both defaults are therefore back to the pre-2026-08-19 shape, so
        /// nothing reaches this code without an explicit environment variable.</para>
        ///
        /// <para>At one chunk per worker the grid is the identity mapping, so this changes nothing for a call
        /// site that has not opted in to a finer split.</para>
        /// </summary>
        internal static readonly bool RegionMajorChunks = Environment.GetEnvironmentVariable(OverfitEnvironment.ParallelRegionMajor) == "1";

        private static int ResolveChunkFactor()
        {
            var raw = Environment.GetEnvironmentVariable(OverfitEnvironment.ParallelChunkFactor);

            if (!int.TryParse(raw, out var factor) || factor < 1)
            {
                // ONE. Four was measured to be 2.1% to 2.9% faster with the region-major layout, and that
                // measurement stands — but shipping it broke the test suite in a way that was never
                // localised: three independent cold runs ended at 176, 188 and 222 of 2748 tests, each
                // reporting "Passed!". A 2% gain is not worth a suite that reports success on 8% of itself,
                // and this project has more guards against that shape of failure than against any other.
                //
                // The switch stays so the measurement is reproducible. See `XC-93`, `XC-95` and the entry in
                // docs/measured-baselines.md for the numbers and for what was NOT established.
                // ONE, because four does not pay in the configuration that ships.
                //
                // Four IS faster, and the measurement stands: three ABAB passes per model, all negative, the
                // 60.9 MB CNN 18.72 to 18.20 ms (-2.8%) and VGG-16 27.63 to 26.70 ms (-3.4%). **But that was
                // measured with the pool pinned to the 16 PHYSICAL cores.** At the shipping pool size —
                // Environment.ProcessorCount, which is 32 logical — the same comparison is +0.0% on the CNN
                // and -1.0% on VGG-16, i.e. nothing.
                //
                // So this waits for the pool size. SMT is already measured to be worth nothing here (16
                // workers 31.22 ms against 32 workers 31.55 ms), and sizing the pool to physical cores would
                // both take that ~1% and unlock this ~3%. The two belong in one change with one measurement.
                return 1;
            }

            return Math.Min(factor, MaxChunkFactor);
        }

        /// <summary>Clears the occupancy totals.</summary>
        public static void ResetOccupancy()
        {
            lock (_gate)
            {
                _occupancyWallTicks = 0;
                _occupancyBusyTicks = 0;
                _occupancyWorstTicks = 0;
                _occupancyWorstWeightedTicks = 0;
                _occupancyFloorWeighted = 0;
                _occupancyFloorBusy = 0;

                for (var i = 0; i < _occupancyBucketCounts.Length; i++)
                {
                    _occupancyBucketCounts[i] = 0;
                    _occupancyBucketBusy[i] = 0;
                }

                _occupancyWorstDispatchRatio = 0;
                _occupancyWorstDispatchChunks = 0;
                _occupancyWorstDispatchBusy = 0;
                _occupancyWorstDispatchItems = 0;
                _occupancyWorstDispatchWidest = 0;
                _occupancyReservedTicks = 0;
                _occupancyChunks = 0;
                _occupancyDispatches = 0;
            }
        }

#pragma warning disable OVERFIT047 // Wall-clock milliseconds and ratios derived from them, for a person to
        // read. The numbers differ between two runs on the same machine, so nothing downstream can match on
        // this text, and the method is called once at the end of a measurement rather than per dispatch.
        /// <summary>
        /// What the fan-outs cost, in the three quantities that separate the three possible causes.
        ///
        /// <list type="bullet">
        ///   <item><description><b>occupancy</b> — executing time over reserved worker time. Low means
        ///   workers were reserved and not working, whatever the reason.</description></item>
        ///   <item><description><b>straggler</b> — the slowest chunk over the average chunk, per dispatch. A
        ///   value of 2.0 means half the reserved worker time was spent waiting at the barrier for one
        ///   chunk, and the fix is the work split.</description></item>
        ///   <item><description><b>overhead</b> — dispatch wall time that is not the slowest chunk. This is
        ///   wake-up, hand-off and join, and it is the part no rebalancing can remove.</description></item>
        ///   <item><description><b>pool use</b> — executing time over the whole pool's time. <b>This is the
        ///   only one of the four that stays meaningful when a fan-out creates more chunks than there are
        ///   workers</b>: occupancy divides by the chunk count and collapses, and overhead subtracts the
        ///   single longest chunk, which stops being the critical path once chunks are small. Measured
        ///   2026-08-19 at four chunks per worker, occupancy read 26.6% and overhead 44.7% for a dispatch
        ///   whose pool use had IMPROVED from 64.6% to 68.3%. Read pool use; read the other three only at
        ///   one chunk per worker. It differs from occupancy exactly when a dispatch creates fewer chunks than there
        ///   are workers, and that difference is the work the split never offered to anybody.</description></item>
        /// </list>
        ///
        /// <para>The three are independent: a balanced split with slow wake-up shows high straggler-free
        /// overhead, while an imbalanced one shows the reverse. Reporting a single "efficiency" number would
        /// merge two causes that call for opposite work.</para>
        /// </summary>
        /// <summary>
        /// The straggler ratio <see cref="OccupancyReport"/> prints, as a number rather than inside a
        /// culture-formatted string. Exists so a test can assert on it without parsing a decimal comma.
        ///
        /// <para>1.00 means every chunk of every dispatch finished in the same time as its siblings. It is
        /// each dispatch's worst chunk weighted by THAT dispatch's chunk count, over total busy time —
        /// which is what makes it cancel to 1.00 for balanced work whatever mix of chunk counts the model
        /// produces. The report reads this property rather than repeating the expression, because two
        /// copies of one formula drift.</para>
        /// </summary>
        internal static double OccupancyStragglerRatio
        {
            get
            {
                lock (_gate)
                {
                    return _occupancyBusyTicks == 0
                        ? 0.0
                        : (double)_occupancyWorstWeightedTicks / _occupancyBusyTicks;
                }
            }
        }

        /// <summary>
        /// The per-dispatch straggler distribution, one line per bucket, plus the single worst fan-out.
        ///
        /// <para><b>Why this exists separately from <see cref="OccupancyReport"/>.</b> The aggregate ratio
        /// answers "how unbalanced", never "where". Those have opposite fixes: an imbalance spread evenly
        /// across every dispatch is a property of how work is split, while one concentrated in a few
        /// dispatches is a property of those layers and the split is fine. Measured 2026-08-21, the 60.9 MB
        /// CNN reports 1.34x aggregate at the shipping default with a uniform-cost reference of 1.02x, so
        /// the imbalance is real — and nothing in the aggregate says which of the 510 dispatches carry
        /// it.</para>
        ///
        /// <para>Both columns are needed. <b>Count</b> is how many fan-outs land in a bucket;
        /// <b>busy</b> is how much worker time they represent. A hundred badly balanced dispatches that
        /// together account for 2% of the run are not the problem, and the count alone would say they
        /// are.</para>
        /// </summary>
        /// <summary>
        /// The largest per-dispatch straggler ratio seen since the last reset, as a number. Exists so a test
        /// can assert on the worst-dispatch tracking without matching a culture-formatted string.
        /// </summary>
        internal static double OccupancyWorstDispatchRatio
        {
            get
            {
                lock (_gate)
                {
                    return _occupancyWorstDispatchRatio;
                }
            }
        }

        public static string OccupancyHistogram()
        {
            lock (_gate)
            {
                if (_occupancyDispatches == 0)
                {
                    return MeasureOccupancy
                        ? "(no fan-out recorded — every call took the inline fast path)"
                        : $"(occupancy not measured — set {OverfitEnvironment.ParallelOccupancy}=1)";
                }

                long totalBusy = 0;

                for (var i = 0; i < _occupancyBucketBusy.Length; i++)
                {
                    totalBusy += _occupancyBucketBusy[i];
                }

                var toMs = 1000.0 / Stopwatch.Frequency;
                var text = new System.Text.StringBuilder();

                text.Append("per-dispatch straggler distribution (")
                    .Append(_occupancyDispatches)
                    .AppendLine(" dispatches)");

                for (var i = 0; i < _occupancyBucketCounts.Length; i++)
                {
                    var label = i == 0
                        ? $"       < {OccupancyBucketEdges[0]:F2}"
                        : i == OccupancyBucketEdges.Length
                            ? $"    >= {OccupancyBucketEdges[^1]:F2}"
                            : $"{OccupancyBucketEdges[i - 1]:F2} - {OccupancyBucketEdges[i]:F2}";

                    var share = totalBusy == 0 ? 0.0 : 100.0 * _occupancyBucketBusy[i] / totalBusy;

                    text.Append("  ")
                        .Append(label.PadLeft(13))
                        .Append("  count ")
                        .Append(_occupancyBucketCounts[i].ToString().PadLeft(5))
                        .Append("   busy ")
                        .Append((_occupancyBucketBusy[i] * toMs).ToString("F1").PadLeft(9))
                        .Append(" ms  ")
                        .Append(share.ToString("F1").PadLeft(5))
                        .AppendLine("%");
                }

                text.Append("  worst dispatch ")
                    .Append(_occupancyWorstDispatchRatio.ToString("F2"))
                    .Append("x over ")
                    .Append(_occupancyWorstDispatchChunks)
                    .Append(" chunks, busy ")
                    .Append((_occupancyWorstDispatchBusy * toMs).ToString("F2"))
                    .Append(" ms");

                if (_occupancyWorstDispatchItems > 0)
                {
                    text.Append(", ")
                        .Append(_occupancyWorstDispatchItems)
                        .Append(" items, widest chunk ")
                        .Append(_occupancyWorstDispatchWidest);
                }

                return text.ToString();
            }
        }

        public static string OccupancyReport()
        {
            lock (_gate)
            {
                if (_occupancyDispatches == 0)
                {
                    return MeasureOccupancy
                        ? "(no fan-out recorded — every call took the inline fast path)"
                        : $"(occupancy not measured — set {OverfitEnvironment.ParallelOccupancy}=1)";
                }

                var toMs = 1000.0 / Stopwatch.Frequency;
                var occupancy = (double)_occupancyBusyTicks / _occupancyReservedTicks;

                // Against the WHOLE pool rather than against the chunks a dispatch happened to create.
                // Occupancy alone understates the loss: a layer that produces 7 chunks on a 16-worker pool
                // reserves 7 and leaves 9 idle, and dividing by 7 scores that as a full house. Measured on
                // the 60.9 MB CNN at 16 workers, mean chunks is 14.4 — so some dispatches do not fill the
                // pool, and only this ratio shows it.
                var poolUse = (double)_occupancyBusyTicks / (_workerCount * (double)_occupancyWallTicks);
                var meanChunks = (double)_occupancyChunks / _occupancyDispatches;
                // `_occupancyWorstTicks` already carries each dispatch's worst chunk WEIGHTED by that
                // dispatch's own chunk count (see RecordOccupancy), so this is a plain ratio. It read
                // `worst * meanChunks / busy` until 2026-08-21, which used the GLOBAL mean against every
                // dispatch and inflated the figure whenever chunk counts varied.
                var straggler = OccupancyStragglerRatio;   // Monitor is re-entrant; one definition.
                var floor = _occupancyFloorBusy == 0 ? 1.0 : _occupancyFloorWeighted / _occupancyFloorBusy;
                var overhead = (double)(_occupancyWallTicks - _occupancyWorstTicks) / _occupancyWallTicks;

                return $"layout {(RegionMajorChunks ? "region-major" : "slice-major")}, "
                    + $"factor {ChunkFactor}, workers {_workerCount}, "
                    + $"dispatches {_occupancyDispatches}, mean chunks {meanChunks:F1}, "
                    + $"wall {_occupancyWallTicks * toMs:F1} ms, busy {_occupancyBusyTicks * toMs:F1} ms | "
                    + $"pool use {100 * poolUse:F1}%, occupancy {100 * occupancy:F1}%, "
                    + $"straggler {straggler:F2}x (uniform-cost reference {floor:F2}x), "
                    + $"overhead {100 * overhead:F1}%";
            }
        }
#pragma warning restore OVERFIT047

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
            void* context,
            int chunksPerWorker = 1)
            => For(rangeStart, rangeEnd, minItemsPerWorker, _workerCount, body, context, chunksPerWorker);

        /// <summary>
        /// Grained <c>For(rangeStart, rangeEnd, minItemsPerWorker, body, context)</c> with an
        /// explicit <paramref name="maxWorkers"/> cap on the chunk (worker) count. Small,
        /// numerous dispatches (single-token decode FFN matmuls) are dispatch-overhead
        /// bound, not bandwidth bound, so fanning them across all cores is a net loss —
        /// the optimum is a handful of workers (see <see cref="DecodeMaxWorkers"/>).
        /// Prefill / training keep the full pool by passing <see cref="WorkerCount"/>.
        /// </summary>
        // chunksPerWorker: how many chunks this fan-out may create per worker. Leave at 1 unless the body
        // is safe with more chunks than there are workers — see ChunkFactor for what "safe" excludes.
        public static void For(
            int rangeStart,
            int rangeEnd,
            int minItemsPerWorker,
            int maxWorkers,
            delegate*<int, int, void*, void> body,
            void* context,
            int chunksPerWorker = 1)
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

            if (CountDispatches)
            {
                Interlocked.Increment(ref _dispatchCount);
            }

            var cap = maxWorkers < 1 ? 1 : Math.Min(maxWorkers, _workerCount);

            // More chunks than workers only when this call site asked for them, bounded by the work itself
            // and by the chunk table. Release(chunkCount - 1) still matches the number of worker chunks
            // exactly, and a worker that finishes one waits again and takes the next — the protocol always
            // supported that, it was only ever handed one chunk each.
            var perWorker = chunksPerWorker < 1 ? 1 : Math.Min(chunksPerWorker, MaxChunkFactor);
            var chunkCount = Math.Min(Math.Min(cap * perWorker, totalWork), _chunks.Length);
            var perChunk = (totalWork + chunkCount - 1) / chunkCount;

            // Taken before the lock on purpose: a dispatch that waits for another one is really waiting, and
            // hiding that would make a serialisation problem look like a fast dispatch.
            var dispatchStarted = MeasureOccupancy ? Stopwatch.GetTimestamp() : 0L;

            lock (_gate)
            {
                _nextChunk.Value = 0;
                _chunkCount = chunkCount;
                _completion.Reset(chunkCount);

                // Plain slicing: chunk i is the (i)th consecutive slice.
                //
                // `XC-95` replaced this with a region-major grid — chunk i as region (i % regions),
                // sub-chunk (i / regions) — on the theory that workers finish in roughly the order they
                // started, so the one that took index k would take k + regions next and continue its own
                // region. **Measured 2026-08-19 and reverted**: on the 60.9 MB CNN it was neutral at two and
                // four chunks per worker and WORSE at eight (+6.5% against the slice-major +4.1%). The
                // approximation is what failed — with sixteen workers claiming dynamically, one that runs 5%
                // slow falls a whole position behind and starts taking somebody else's region, so nothing
                // stays contiguous. Locality needs a worker PINNED to a region, which is a change to the
                // claim protocol rather than to this loop.
                // The chunk count is rounded to an exact regions x subChunks grid for the region-major
                // layout. Without that, a count that is not a multiple of the region count leaves some
                // regions without their last sub-chunk and part of the range is never executed — silently,
                // because every chunk still signals and the dispatch still completes. Caught by
                // OverfitParallelChunkGridTests before it ever ran.
                var regions = RegionMajorChunks ? Math.Min(cap, chunkCount) : chunkCount;
                var subChunks = Math.Max(1, chunkCount / regions);

                if (RegionMajorChunks)
                {
                    chunkCount = regions * subChunks;
                    _chunkCount = chunkCount;
                    _completion.Reset(chunkCount);
                }

                var regionWork = (totalWork + regions - 1) / regions;
                var subWork = (regionWork + subChunks - 1) / subChunks;

                for (var i = 0; i < chunkCount; i++)
                {
                    var chunkStart = (long)rangeStart + ((long)i * perChunk);
                    var chunkEnd = Math.Min(chunkStart + perChunk, rangeEnd);

                    if (RegionMajorChunks)
                    {
                        var regionStart = (long)rangeStart + ((long)(i % regions) * regionWork);
                        var regionEnd = Math.Min(regionStart + regionWork, rangeEnd);

                        chunkStart = Math.Min(regionStart + ((long)(i / regions) * subWork), regionEnd);
                        chunkEnd = Math.Min(chunkStart + subWork, regionEnd);
                    }

                    _chunks[i].Start = (int)chunkStart;
                    _chunks[i].End = (int)chunkEnd;
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

                if (MeasureOccupancy)
                {
                    RecordOccupancy(chunkCount, Stopwatch.GetTimestamp() - dispatchStarted);
                }

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
        /// sized to the cap, so there are no idle spinners. Serialised by <c>_decodeGate</c>.
        ///
        /// <para><b>Its claim protocol is NOT the main pool's</b> — this line used to say it was, and the
        /// difference is why only this pool needed the 2026-08-14 fix. The main pool commits to completion
        /// before a worker claims anything: one <c>SemaphoreSlim</c> token is consumed per chunk, so a
        /// straggler cannot take work belonging to a later dispatch because there is no token for it. This
        /// pool has no such token — workers poll a generation counter — so the claim itself has to carry
        /// the generation it belongs to. See <see cref="TryClaimDecodeChunk"/>.</para>
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
                var generation = _decodeGen + 1;

                // Resetting the completion counter BEFORE the dispatch is published looks wrong and is
                // not, but it is safe only as a CONSEQUENCE of the claim invariant, so the reasoning is
                // recorded here rather than left to be re-derived. _decodeRemaining is decremented exactly
                // once per SUCCESSFUL claim (ExecuteDecodeChunk's finally). Claims tagged G total exactly
                // chunkCount_G, because the bound they test against is G's own and travels in G's word.
                // So when G's dispatcher observed _decodeRemaining == 0 below, every G execution had
                // already decremented and no further G claim can succeed — no G decrement can survive into
                // G+1 and land on this reset value. That argument holds ONLY while the claim's bound is
                // generation-correct: under the pre-`XC-50` shape the extra claim a straggler could win is
                // precisely what broke it.
                //
                // WHY THAT PREMISE HOLDS ACROSS TWO DIFFERENT ForDecode CALLS — the step the paragraph
                // above assumes and never states: `_decodeGate` serialises dispatches end to end. G's
                // dispatcher takes the lock, publishes, drains, and only LEAVES it after its own spin
                // below has observed _decodeRemaining == 0; G+1's dispatcher cannot reach this line until
                // then, because it is still waiting on the same lock. So "G observed zero" is not a
                // property of some earlier call in the abstract — it is a fact established before this
                // reset can execute at all. Remove the lock, or move the completion spin outside it, and
                // this reset stops being safe.
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

                // Publish the claim word first, then the generation. Both writes are releases, so the
                // descriptors above are visible to anyone who acquires the word.
                //
                // THIS COMMENT USED TO ARGUE THAT A WORKER "only starts claiming once it has seen the new
                // generation, so it can never observe a claim word that still belongs to the previous
                // dispatch". That reasoning is about the wrong worker and it is what hid `XC-50`. The hole
                // belongs to a straggler still draining the OLD generation, which never looks at
                // _decodeGen again — it is holding G in a local and re-entering the claim. Nothing about
                // publication order restrains it.
                //
                // What restrains it is that everything its claim tests is in the one word: tag, bound and
                // next index are written here by a single store, so it either sees G's word (its own tag,
                // its own exhausted bound) or G+1's (tag mismatch). While the bound lived in a separate
                // field published earlier, a third state existed — G's tag with G+1's bound — and a
                // straggler drove straight through it.
                DecodeChunkClaim.Publish(ref _decodeClaim.Value, chunkCount, generation);
                Volatile.Write(ref _decodeGen, generation);

                // Wake parked workers. Unconditional — see _decodeParkLock for why a "is anybody parked"
                // fast path cannot be made correct without a full fence that costs the same as the lock.
                lock (_decodeParkLock)
                {
                    Monitor.PulseAll(_decodeParkLock);
                }

                // Calling thread participates — greedy drain (safe under _decodeGate).
                // BOUND: the chunk count carried in the claim word — this dispatch's `chunkCount`, just
                // published above. Every successful claim advances the word's index field, so the loop
                // runs at most chunkCount times before TryClaimDecodeChunk returns false. No OVERFIT023
                // suppression needed any more — the condition is no longer `true`.
                while (TryClaimDecodeChunk(generation, out var index))
                {
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

        /// <summary>
        /// Binds this class's claim word to the claim protocol, which lives in
        /// <see cref="DecodeChunkClaim.TryClaim"/> — see there for why the generation, the chunk count and
        /// the index share one word.
        ///
        /// <para>The word is the only thing that crosses: nothing else about the dispatch is passed, and
        /// nothing else could be, since this class's dispatch state is private to it. Any future parameter
        /// added here is `XC-50` returning.</para>
        /// </summary>
        private static bool TryClaimDecodeChunk(long generation, out int index)
        {
            return DecodeChunkClaim.TryClaim(ref _decodeClaim.Value, generation, out index);
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
        // OVERFIT040 for the two worker loops below (DecodeWorkerLoop and WorkerLoop).
        //
        // THE CONSTRAINT: these run on DEDICATED BACKGROUND THREADS this class creates and owns for the
        // process's lifetime — not on thread-pool threads. Blocking one is what it is FOR: a parked worker
        // holding its own thread is how it resumes in nanoseconds when the next generation is dispatched.
        // `WaitAsync` would hand the continuation back to the thread pool and reintroduce exactly the
        // scheduling latency this pool exists to avoid.
        //
        // WHAT IT IS WORTH, measured — re-audited 2026-08-14 (`PB-12`) on a Ryzen 9 9950X3D (32 logical),
        // .NET 10.0.8, Release, HEAD e21e7c3, single-process ABAB, 183 dispatches over a 4096 range:
        // the decode pool runs 617 us / 0 B against `Parallel.For` at 1502 us / 505 KB when that arm is
        // capped at DecodeMaxWorkers — which is the arm OVERFIT_DECODE_POOL=0 actually falls back to — and
        // 2223 us / 862 KB uncapped. So 2.4x against the comparison the product makes and 3.6x against the
        // one it no longer makes anywhere; ~+25% end-to-end on Qwen3-0.6B. The `455 us vs 2059 us = 4.5x`
        // published here until 2026-08-14 quoted the UNCAPPED pair and is retired — see
        // docs/measured-baselines.md, which is the canonical copy. Zero allocation instead of hundreds of KB
        // (862 KB - 1.13 MB, shape-dependent) is the other half, and together they are the reason the answer
        // here is not "await it".
        //
        // WHAT IS GIVEN UP: one OS thread per pool slot, parked, for as long as the process runs. The pool is
        // sized to the decode cap (<= core count), which is the bound that makes that acceptable.
#pragma warning disable OVERFIT040
        private static void DecodeWorkerLoop()
        {
            var seen = 0L;

            // BOUND: none by design — this is a daemon worker that parks until the process exits. It runs on
            // a background thread (IsBackground = true), so it cannot keep the process alive; the "hang" this
            // rule guards against is a foreground loop that never yields a result, which this is not.
#pragma warning disable OVERFIT023
            while (true)
#pragma warning restore OVERFIT023
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

                    // Budget exhausted — park until the generation moves. The wait is on the CONDITION,
                    // re-tested under the lock after every wake, so a surplus pulse costs one re-check and
                    // a missed one is impossible: the dispatcher publishes the generation and then pulses
                    // while holding this same lock. The counted-semaphore version leaked a token every
                    // time a worker registered and then skipped its wait, which stopped the pool sleeping
                    // at all — 14.85 effective cores on an idle box.
                    lock (_decodeParkLock)
                    {
                        // BOUND: the generation is monotonic and every dispatch pulses this lock, so the
                        // predicate is false after at most one dispatch.
                        while (Volatile.Read(ref _decodeGen) == seen)
                        {
                            Monitor.Wait(_decodeParkLock);
                        }
                    }

                    spins = 0;
                }

                seen = gen;

                // Drain this generation greedily. Claims are generation-checked, so a worker that arrives
                // late simply finds nothing to take rather than stealing from the dispatch that replaced it.
                while (TryClaimDecodeChunk(gen, out var index))
                {
                    ExecuteDecodeChunk(index);
                }
            }
        }
#pragma warning restore OVERFIT040

        /// <summary>
        /// Runs one chunk's body, capturing any thrown exception into the chunk
        /// descriptor and signalling completion. Shared by the background
        /// workers and the calling thread (caller participation).
        /// </summary>
        private static void ExecuteChunk(int index)
        {
            // Read once into a local: the flag is checked before the timestamp so the happy path is a
            // predictable, never-taken branch rather than a clock read.
            var measured = MeasureOccupancy;
            var started = measured ? Stopwatch.GetTimestamp() : 0L;

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
                if (measured)
                {
                    // Before Signal(), so the caller reading it after Wait() sees the write.
                    _chunks[index].BusyTicks = Stopwatch.GetTimestamp() - started;
                }

                _completion.Signal();
            }
        }

        // OVERFIT040: same constraint and same measurement as DecodeWorkerLoop above — a dedicated background
        // thread this class owns, parked on its own semaphore, where blocking IS the design.
#pragma warning disable OVERFIT040
        private static void WorkerLoop()
        {
            // BOUND: none by design — daemon worker, parks on _startSemaphore until the process exits. Runs on
            // a background thread, so it never blocks shutdown.
#pragma warning disable OVERFIT023
            while (true)
#pragma warning restore OVERFIT023
            {
                _startSemaphore.Wait();

                // Claim a unique chunk index. Interlocked.Increment is a
                // full fence — pairs with the semaphore release so the
                // descriptor reads below see the dispatcher's writes.
                var index = Interlocked.Increment(ref _nextChunk.Value) - 1;

                if (index >= _chunkCount)
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
                    continue;
                }

                ExecuteChunk(index);
            }
        }
#pragma warning restore OVERFIT040

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

            // Written by the worker before it signals completion, so the release fence of Signal() is what
            // publishes it to the caller. Only touched when MeasureOccupancy is on; it takes one of the
            // padding slots rather than growing the struct, so the 64-byte line is preserved either way.
            public long BusyTicks;                               // 8

            // 40 bytes used; pad to 64-byte cache line.
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

        /// <summary>
        /// Cache-line-padded 64-bit work claim: the dispatch generation in the high 32 bits, the chunk
        /// count in the next 16, the next unclaimed chunk index in the low 16 (layout owned by
        /// <see cref="DecodeChunkClaim"/>). Padded for the same reason as <see cref="PaddedCounter"/> —
        /// every worker hammers it with a compare-and-swap.
        ///
        /// <para>One word rather than three fields so that <see cref="TryClaimDecodeChunk"/> can verify the
        /// generation, test the bound and take the index in a single atomic operation; see
        /// <see cref="DecodeChunkClaim"/> for what went wrong each time one of the three was read from
        /// somewhere else.</para>
        /// </summary>
        [StructLayout(LayoutKind.Explicit, Size = 128)]
        private struct PaddedClaim
        {
            [FieldOffset(64)]
            public long Value;
        }
    }
}
