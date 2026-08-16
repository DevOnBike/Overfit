// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Idle-burn gate for the decode spin-pool (spin-then-park, 2026-06-11): after a decode finishes, the
    /// pool must PARK — a serving container at rest must not spin cores (field report: 100% CPU at idle on
    /// a 16-core laptop with the pre-park pool). Decodes a few tokens to wake the pool, then sleeps and
    /// asserts the pool is quiet over the idle window. [LongFact] — needs C:\qwen3-06b.
    ///
    /// <para><b>The instrument measures the POOL'S OWN THREADS, not the process</b> (<c>TG-T13</c>,
    /// 2026-08-16). It used to read <see cref="Process.TotalProcessorTime"/>, which counts every thread in
    /// the process: in an area run that measured <b>14.57 effective cores on a 32-logical-core box</b> —
    /// real CPU, none of it the pool's. <c>XC-54</c> fixed that by making the window quiet
    /// (<see cref="ExclusiveProcessMeasurementCollection"/>) and adding a canary, which made process CPU a
    /// fair proxy but left the test unable to tell <i>the pool is spinning</i> from <i>something else in
    /// this process is busy</i>. It now identifies the decode workers' OS thread ids and sums
    /// <see cref="ProcessThread.TotalProcessorTime"/> over exactly those, so the number is the pool's.</para>
    ///
    /// <para><b>How the pool's threads are identified — by running on them, not by matching an OS thread
    /// name.</b> <see cref="ProcessThread"/> carries no name, and the managed <see cref="Thread.Name"/> is
    /// not the OS thread name on every platform, so a name lookup from outside is not available. Instead
    /// this class dispatches its own <see cref="OverfitParallel.ForDecode"/> whose body runs <i>on</i> each
    /// worker and reads <see cref="Thread.CurrentThread"/>'s managed name (assigned at
    /// <c>OverfitParallel.cs:460</c> as <c>OverfitDecode-{i}</c>) together with the OS thread id. Only
    /// bodies that ran on a thread carrying that name are kept, which excludes both the calling thread —
    /// <c>ForDecode</c> makes the caller participate, and it was observed doing so — and the general
    /// <c>OverfitParallel-{i}</c> pool, which is a different pool and not this test's subject. The bodies
    /// hold a bounded barrier so one worker cannot drain every chunk, which is what spreads a dispatch
    /// across distinct threads; the rounds repeat until all <see cref="OverfitParallel.DecodePoolSize"/>
    /// workers have been seen. <b>Measured on the dev box: all 10 identified in round 0.</b></para>
    ///
    /// <para><b>The canary is gone, deliberately, and the three-valued verdict became a capability
    /// guard.</b> The canary existed because a process-wide instrument cannot separate subject from
    /// environment; a scoped one can, so there is nothing left for it to detect — and keeping it would have
    /// been actively harmful, because the only thing that can now make a pre-decode window busy is the pool
    /// itself failing to park, which is a RED and must never be reported as "could not measure". What
    /// remains three-valued is whether the test can run at all: no decode pool
    /// (<c>OVERFIT_DECODE_POOL=0</c> or a pool of one), not Windows, or fewer than
    /// <c>DecodePoolSize</c> workers identified, all skip with the reason. <see cref="ProcessThread"/>
    /// per-thread CPU is unsupported on macOS and <see cref="ProcessThread.ThreadState"/> is Windows-only,
    /// and this test needs a multi-GB GGUF that lives on the Windows dev box, so the platform guard costs
    /// nothing real and is stated rather than assumed.</para>
    ///
    /// <para><b>Why <see cref="ExclusiveProcessMeasurementCollection"/> is still needed after scoping.</b>
    /// Its job changed: it no longer buys a quiet CPU counter, it stops another test dispatching decode
    /// work through the same static pool during this window, which would be a false RED against threads
    /// that are legitimately busy. Scoping does not help there — the CPU is the pool's, it is simply not
    /// this test's.</para>
    ///
    /// <para><b>Two assertions, because CPU rate alone can be starved into a false GREEN.</b> External load
    /// does not enter this counter, but it biases the result <b>downwards</b>: a genuinely spinning pool
    /// competing with other processes accumulates less CPU per wall second, so a heavily loaded box can
    /// make a spinning pool look quiet. <b>Scoping does not remove that bias</b> — it is a property of
    /// measuring a rate that the scheduler caps, not of the counter's scope. The second assertion is
    /// load-invariant instead of rate-based: sampling <see cref="ProcessThread.ThreadState"/> across the
    /// window, a parked worker sits in <see cref="System.Diagnostics.ThreadState.Wait"/> however busy the
    /// box is, while a spinner starved of cores sits in <c>Running</c> or <c>Ready</c> and never in
    /// <c>Wait</c>.</para>
    ///
    /// <para><b>That is measured, not reasoned</b> (2026-08-16, 32 logical, Ryzen 9 9950X3D). Park mutation
    /// (<c>OverfitParallel.cs:868</c>, the spin budget replaced by <c>int.MaxValue</c>) on an idle box:
    /// <b>9.97 cores, 0.0% parked</b>. The same mutation with <b>96 external spinner processes</b> — 3x
    /// oversubscription: <b>4.35 cores, 0.0% parked</b>. So the downward bias on the rate is real and worth
    /// 2.3x at that load, while the parked fraction did not move at all. Nothing here shows the rate
    /// crossing the 1.00 gate, so the false GREEN remains a possibility on a box loaded harder than this
    /// one — that is exactly the case the second assertion covers, and it is why it exists rather than
    /// being folded into a wider tolerance on the first.</para>
    /// </summary>
    [Collection(ExclusiveProcessMeasurementCollection.Name)]
    public sealed partial class DecodePoolIdleBurnTests
    {
        private const string Path = @"C:\qwen3-06b\Qwen3-0.6B-Q4_K_M.gguf";

        /// <summary>Length of the idle measurement window.</summary>
        private const int WindowMilliseconds = 3000;

        /// <summary>
        /// The gate on CPU rate. A parked pool reads ~0.05 cores on the dev box — ten threads times one
        /// 15.625 ms Windows accounting tick over three seconds, i.e. the quantisation floor of the
        /// instrument rather than work. The pre-park pure-spin pool reads ~10, and 4.35 with the box
        /// oversubscribed 3x. Nothing measured so far sits near 1.0 from either side.
        /// </summary>
        private const double IdleBurnLimitCores = 1.0;

        /// <summary>
        /// The gate on parked-state samples: the fraction of (pool thread, sample) observations that must
        /// find the thread waiting. Measured on the dev box: <b>1.000 parked</b> for a healthy pool,
        /// <b>0.000</b> under the spin mutation, and <b>0.000</b> under the spin mutation with the box
        /// oversubscribed 3x, so this floor sits clear of every arm that has been run. It
        /// is not 1.0 because a sample taken while the runtime has the thread suspended for a GC, or while
        /// a worker is between its wait and its re-check, is legitimate and rare.
        /// </summary>
        private const double ParkedSampleFloor = 0.90;

        /// <summary>
        /// Gap between <see cref="ProcessThread.ThreadState"/> samples over the idle window. It sets how
        /// many samples the window yields, not the verdict: the assertion is on the FRACTION parked, so a
        /// loaded box that starves the sampling loop simply produces fewer observations. Measured: 28
        /// samples idle against 14 under 3x oversubscription, with the fraction unchanged at both.
        /// </summary>
        private const int SampleIntervalMilliseconds = 100;

        /// <summary>
        /// Managed name prefix given to the decode pool's workers at <c>OverfitParallel.cs:460</c>. Read
        /// back on the worker itself, so this is the string the pool assigned and not an OS thread name.
        /// </summary>
        private const string DecodeThreadNamePrefix = "OverfitDecode-";

        /// <summary>
        /// How many identification dispatches to run before giving up. One sufficed on the dev box; the
        /// loop exists because which worker claims which chunk is a schedule, and a worker still parked
        /// from an earlier dispatch can miss a round.
        /// </summary>
        private const int IdentificationRounds = 8;

        /// <summary>
        /// Bound on the identification barrier. It is a backstop, not a timing assertion: nothing about the
        /// verdict depends on how long a round takes, only on which threads it reached.
        /// </summary>
        private const int BarrierTimeoutMilliseconds = 2000;

        private readonly ITestOutputHelper _out;

        public DecodePoolIdleBurnTests(ITestOutputHelper output) => _out = output;

        [LibraryImport("kernel32.dll")]
        private static partial uint GetCurrentThreadId();

        [ModelFact(Path, "4s")]
        public void Pool_Parks_WhenIdle()
        {
            if (!OperatingSystem.IsWindows())
            {
                Assert.Skip(
                    "cannot measure: per-thread CPU and thread state are read through ProcessThread, which "
                    + "is Windows-only for ThreadState and unsupported on macOS for TotalProcessorTime.");
                return;
            }

            Assert.SkipWhen(
                !OverfitParallel.DecodePoolEnabled || OverfitParallel.DecodePoolSize < 2,
                $"cannot measure: there is no decode pool to observe (enabled="
                + $"{OverfitParallel.DecodePoolEnabled}, size={OverfitParallel.DecodePoolSize}). ForDecode "
                + "delegates elsewhere in that configuration, so this test has no subject.");

            var poolThreadIds = IdentifyDecodePoolThreads(out var rounds);
            _out.WriteLine(
                $"identified {poolThreadIds.Count} of {OverfitParallel.DecodePoolSize} decode workers in "
                + $"{rounds} round(s): {string.Join(", ", poolThreadIds)}");

            Assert.SkipWhen(
                poolThreadIds.Count != OverfitParallel.DecodePoolSize,
                $"cannot measure: identified {poolThreadIds.Count} of "
                + $"{OverfitParallel.DecodePoolSize} decode workers over {rounds} dispatches. Measuring a "
                + "subset would under-report the pool's burn, which is the direction that produces a false "
                + "pass, so this is neither a pass nor a failure.");

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);
            using var session = engine.CreateSession(128);
            session.Reset(tok.Encode("Hello"));
            var sampling = SamplingOptions.Greedy;
            for (var i = 0; i < 8; i++)
            {
                session.GenerateNextToken(in sampling);
            }   // pool is hot now

            var window = MeasureDecodePoolIdleWindow(poolThreadIds);

            _out.WriteLine(
                $"idle window {window.WindowSeconds:F1}s over {window.ThreadsObserved} pool threads: "
                + $"{window.EffectiveCores:F2} effective cores, parked in "
                + $"{window.ParkedFraction:P1} of {window.Observations} (thread, sample) observations "
                + $"from {window.Samples} samples. The whole process read {window.ProcessCores:F2} cores "
                + "over the same window — reported for comparison, not asserted on.");

            Assert.SkipWhen(
                window.ThreadsObserved != poolThreadIds.Count,
                $"cannot measure: {window.ThreadsObserved} of {poolThreadIds.Count} identified decode "
                + "workers were still present in Process.Threads at the end of the window.");

            Assert.True(
                window.EffectiveCores < IdleBurnLimitCores,
                $"decode pool is burning CPU at idle: {window.EffectiveCores:F2} effective cores summed "
                + $"over its own {window.ThreadsObserved} worker threads (expected ~0 after spin-then-park; "
                + $"limit {IdleBurnLimitCores:F2})");

            Assert.True(
                window.ParkedFraction >= ParkedSampleFloor,
                $"decode pool is not parked at idle: its worker threads were waiting in only "
                + $"{window.ParkedFraction:P1} of {window.Observations} samples (floor "
                + $"{ParkedSampleFloor:P0}). This assertion is the one a loaded box cannot starve into "
                + "silence — a spinner denied cores is Running or Ready, never Waiting.");
        }

        /// <summary>
        /// The measured idle window: what the pool's own threads did, and nothing else's.
        /// </summary>
        /// <param name="ProcessCores">
        /// What the pre-<c>TG-T13</c> instrument would have read over the same window: the whole process.
        /// <b>Reported, never asserted</b> — it is here so every run carries the gap between the two
        /// instruments as data rather than as this class's docstring claiming one. In an isolated run the
        /// two are within noise of each other; the difference is what a neighbour costs.
        /// </param>
        private readonly record struct IdleWindow(
            double EffectiveCores,
            double ProcessCores,
            double WindowSeconds,
            double ParkedFraction,
            int Observations,
            int Samples,
            int ThreadsObserved);

        /// <summary>
        /// Sums CPU over the identified decode workers across one idle window, and samples how often those
        /// threads are found waiting. Both are per-thread reads: nothing here observes the process.
        /// </summary>
        /// <param name="poolThreadIds">OS thread ids of the decode pool's workers.</param>
        /// <returns>The window's CPU rate in effective cores plus the parked-sample fraction.</returns>
        [SupportedOSPlatform("windows")]
        private static IdleWindow MeasureDecodePoolIdleWindow(HashSet<int> poolThreadIds)
        {
            using var process = Process.GetCurrentProcess();
            process.Refresh();

            var processBefore = process.TotalProcessorTime;
            var before = new Dictionary<int, TimeSpan>(poolThreadIds.Count);
            foreach (ProcessThread thread in process.Threads)
            {
                if (poolThreadIds.Contains(thread.Id))
                {
                    before[thread.Id] = thread.TotalProcessorTime;
                }
            }

            var parked = 0;
            var observations = 0;
            var samples = 0;
            var elapsed = Stopwatch.StartNew();

            while (elapsed.ElapsedMilliseconds < WindowMilliseconds)
            {
                Thread.Sleep(SampleIntervalMilliseconds);

                // Refresh() is what re-reads the thread table; without it every sample would repeat the
                // first one. Its cost lands on THIS thread, which is not one of the subjects — the one
                // thing scoping buys that a process-wide counter could never have.
                process.Refresh();
                samples++;

                foreach (ProcessThread thread in process.Threads)
                {
                    if (!poolThreadIds.Contains(thread.Id))
                    {
                        continue;
                    }

                    observations++;

                    if (thread.ThreadState == System.Diagnostics.ThreadState.Wait)
                    {
                        parked++;
                    }
                }
            }

            elapsed.Stop();
            process.Refresh();

            var processSeconds = (process.TotalProcessorTime - processBefore).TotalSeconds;
            var cpuSeconds = 0.0;
            var observed = 0;

            foreach (ProcessThread thread in process.Threads)
            {
                if (!before.TryGetValue(thread.Id, out var start))
                {
                    continue;
                }

                observed++;
                cpuSeconds += (thread.TotalProcessorTime - start).TotalSeconds;
            }

            var windowSeconds = elapsed.Elapsed.TotalSeconds;

            return new IdleWindow(
                cpuSeconds / windowSeconds,
                processSeconds / windowSeconds,
                windowSeconds,
                observations == 0 ? 0.0 : (double)parked / observations,
                observations,
                samples,
                observed);
        }

        /// <summary>
        /// Context for <see cref="IdentifyBody"/>. Pointers into pinned test-owned arrays; no field of this
        /// struct outlives the <c>fixed</c> block that publishes it, which is the same discipline
        /// production uses.
        /// </summary>
        private unsafe struct IdentifyContext
        {
            public int* Arrived;
            public uint* ThreadIds;
            public int ChunkCount;
            public int Capacity;
        }

        /// <summary>
        /// Records the OS thread id of the decode worker it runs on, then holds until every chunk of this
        /// dispatch has arrived so that a single worker cannot drain them all.
        /// </summary>
        private static unsafe void IdentifyBody(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<IdentifyContext>(context);
            var slot = Interlocked.Increment(ref *ctx.Arrived) - 1;
            var name = Thread.CurrentThread.Name;

            if (slot < ctx.Capacity
                && name != null
                && name.StartsWith(DecodeThreadNamePrefix, StringComparison.Ordinal))
            {
                ctx.ThreadIds[slot] = GetCurrentThreadId();
            }

            // The barrier is what forces the dispatch across distinct threads: a worker holding chunk i
            // cannot claim chunk i+1. Bounded, so a pool that cannot field ChunkCount workers still
            // returns — the caller then simply identifies fewer of them and the test says so.
            var deadline = Stopwatch.StartNew();

            while (Volatile.Read(ref *ctx.Arrived) < ctx.ChunkCount
                   && deadline.ElapsedMilliseconds < BarrierTimeoutMilliseconds)
            {
                Thread.SpinWait(64);
            }
        }

        /// <summary>
        /// Runs identification dispatches until every decode worker has been seen, or the rounds run out.
        /// </summary>
        /// <param name="rounds">How many dispatches were needed.</param>
        /// <returns>OS thread ids of the decode pool's workers, which may be short of the pool size.</returns>
        private static unsafe HashSet<int> IdentifyDecodePoolThreads(out int rounds)
        {
            var poolSize = OverfitParallel.DecodePoolSize;
            var identified = new HashSet<int>(poolSize);

            // One slot per chunk plus one for the participating caller, whose body also arrives.
            var captured = new uint[poolSize + 1];
            var arrived = new int[1];

            rounds = 0;

            for (var round = 0; round < IdentificationRounds; round++)
            {
                rounds++;
                arrived[0] = 0;
                captured.AsSpan().Clear();

                fixed (int* arrivedPtr = arrived)
                fixed (uint* capturedPtr = captured)
                {
                    var context = new IdentifyContext
                    {
                        Arrived = arrivedPtr,
                        ThreadIds = capturedPtr,
                        ChunkCount = poolSize,
                        Capacity = captured.Length,
                    };

                    OverfitParallel.ForDecode(0, poolSize, &IdentifyBody, &context);
                }

                foreach (var id in captured)
                {
                    if (id != 0)
                    {
                        identified.Add((int)id);
                    }
                }

                if (identified.Count == poolSize)
                {
                    break;
                }
            }

            return identified;
        }
    }
}
