// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// Wake gate for the decode spin-pool (<c>XC-69</c>): a worker that has PARKED must be woken by the
    /// next dispatch. <c>DecodePoolIdleBurnTests.Pool_Parks_WhenIdle</c> covers the other half — that an
    /// idle pool parks at all — and this class deliberately does not repeat it.
    ///
    /// <para><b>The defect class this exists for is silent.</b> Delete
    /// <c>Monitor.PulseAll(_decodeParkLock)</c> (<c>OverfitParallel.cs:756</c>) and parked workers are never
    /// woken again: the dispatch still completes, because the calling thread drains greedily; the output is
    /// still correct; nothing throws and nothing hangs. The pool simply degrades to one thread. No
    /// throughput assertion in this repository is watching, which is the same shape as <c>PB-12</c>, where a
    /// synchronisation change on this path survived the headline numbers for weeks.</para>
    ///
    /// <para><b>Why the obvious assertion is NOT admissible, and what replaces it.</b> "After idling the
    /// pool, a subsequent dispatch's chunks are executed by pool threads" asserts the outcome of a race:
    /// <c>ForDecode</c> makes the caller participate (<c>OverfitParallel.cs:764</c>), so a legal schedule
    /// exists in which the calling thread claims every chunk before any worker is scheduled — with the
    /// pulse fully intact. A test asserting otherwise is <c>TG-T12</c> again: green on an idle box, red on a
    /// loaded one. This class instead makes worker participation <b>necessary for the dispatch to
    /// terminate</b>. Every chunk body blocks until a body has entered on a thread named
    /// <c>OverfitDecode-*</c>; the calling thread therefore blocks inside its first chunk and cannot claim
    /// another, so at least one chunk of this dispatch stays claimable by a worker and by nobody else.</para>
    ///
    /// <para><b>The schedule-invariance argument, in full.</b> Let the dispatch publish
    /// <c>chunkCount == DecodePoolSize &gt;= 2</c>. (i) The calling thread either claims no chunk — in which
    /// case workers claimed them all and the property holds outright — or claims one and blocks in its body.
    /// (ii) Blocked, it claims nothing further, so <c>chunkCount - 1 &gt;= 1</c> chunks remain unclaimed.
    /// (iii) <c>_decodeGen</c> has been advanced before the pulse, and every worker was observed parked, so
    /// each is inside <c>Monitor.Wait</c> and every one of them is pulsed. (iv) Under fairness — every
    /// runnable thread eventually runs, which is the assumption any concurrent program needs to terminate
    /// and is NOT a timing assumption — some worker re-acquires the lock, re-tests the predicate, finds the
    /// generation moved, and calls <c>TryClaimDecodeChunk</c>, which succeeds while unclaimed chunks remain.
    /// (v) It enters a body, and the property is established. Nothing in that chain depends on which thread
    /// wins any race, on how many workers wake, or on how long anything takes.</para>
    ///
    /// <para><b>The one wall-clock number here is a backstop, not the assertion</b>, and the reason it is
    /// not <c>TG-T12</c> is that there is no continuum between the two arms: a healthy pool satisfies the
    /// predicate in microseconds, and a pool whose pulse is gone never satisfies it — <c>Monitor.Wait</c>
    /// with no timeout returns only on a pulse, so the failure is permanent rather than slow.
    /// <see cref="WakeDeadlineMilliseconds"/> sits seven orders of magnitude above the healthy path, and on
    /// expiry the body sets an abort flag and returns rather than throwing or hanging — a hang on this path
    /// costs the machine-wide measurement mutex, which orphaned a run for 37 minutes during
    /// <c>XC-50</c>.</para>
    ///
    /// <para><b>Parking is established by observation, not by sleeping long enough.</b> A worker parks only
    /// after <c>DecodeSpinBudgetIterations</c> (~1-2 ms of <c>SpinWait(32)</c>), and on a loaded box that is
    /// longer in wall time, so a fixed sleep would be exactly the assumption this class refuses to make
    /// elsewhere. Instead the workers are identified by running on them, their managed
    /// <see cref="Thread"/> objects are kept, and <see cref="Thread.ThreadState"/> is polled until every one
    /// of them reports <see cref="System.Threading.ThreadState.WaitSleepJoin"/>. That state is
    /// load-invariant: a spinner starved of cores is Running, never Waiting (measured under <c>TG-T13</c> —
    /// 100.0% parked healthy, 0.0% under the park mutation, 0.0% under the park mutation with the box
    /// oversubscribed 3x). If the poll runs out, this test <b>skips with the observed states</b> rather than
    /// failing: whether the pool parks is another test's subject, and a wake test on a pool that never
    /// parked has no subject.</para>
    ///
    /// <para><b>What this does NOT cover, measured rather than reasoned (2026-08-16).</b> Two mutations were
    /// run. Removing the pulse entirely reddens this test in 30 s with its named message — and reddens
    /// <b>nothing else</b>: the rest of the fast suite was 0 failures of 2659 under that mutation, so this
    /// is added coverage and not moved coverage. But <b>publishing the generation AFTER the pulse instead of
    /// before it — the classic lost wakeup, and the ordering step (iii) above leans on — leaves this test
    /// GREEN</b> (9 of 10 chunks still ran on distinct workers). The reason is that <c>Monitor.PulseAll</c>
    /// fires inside the lock, so a woken waiter cannot return from <c>Wait</c> until the dispatcher releases
    /// it, by which point the very next instruction has already written the generation. The hole is real but
    /// its window is a few instructions wide, so it is a race that essentially never fires rather than a
    /// permanent failure — which is exactly why no schedule-invariant assertion can catch it, this one
    /// included. <b>The publication order is therefore held by the comment at <c>OverfitParallel.cs:734</c>
    /// and by nothing executable.</b></para>
    ///
    /// <para><b>Membership of <see cref="ExclusiveProcessMeasurementCollection"/></b> is for the reason that
    /// collection documents for a scoped test: nothing else may drive the same static pool between the
    /// moment parking is observed and the moment the dispatch is published. A neighbour's
    /// <c>ForDecode</c> in that window would wake the workers for free and make this test pass with the
    /// pulse removed.</para>
    /// </summary>
    [Collection(ExclusiveProcessMeasurementCollection.Name)]
    public sealed unsafe class DecodePoolWakeTests
    {
        /// <summary>
        /// Managed name prefix the pool assigns its workers at <c>OverfitParallel.cs:460</c>. Read back on
        /// the worker itself, so it is the string the pool assigned and not an OS thread name.
        /// </summary>
        private const string DecodeThreadNamePrefix = "OverfitDecode-";

        /// <summary>
        /// How many identification dispatches to run before giving up. One sufficed on the dev box; the loop
        /// exists because which worker claims which chunk is a schedule.
        /// </summary>
        private const int IdentificationRounds = 8;

        /// <summary>
        /// Bound on the identification barrier. A backstop, not a timing assertion: nothing about the
        /// verdict depends on how long a round takes, only on which threads it reached.
        /// </summary>
        private const int IdentificationBarrierTimeoutMilliseconds = 2000;

        /// <summary>Gap between <see cref="Thread.ThreadState"/> polls while waiting for the pool to park.</summary>
        private const int ParkPollIntervalMilliseconds = 10;

        /// <summary>
        /// Bound on waiting for every identified worker to park. The spin budget is ~1-2 ms of CPU, so this
        /// is three orders of magnitude of slack; running out is reported as "cannot measure", never as a
        /// wake failure.
        /// </summary>
        private const int ParkPollBudgetMilliseconds = 5000;

        /// <summary>
        /// Bound on the wake rendezvous. Healthy: microseconds. Pulse removed: never — so any value that
        /// clears scheduling noise separates the arms, and this one clears it by ~7 orders of magnitude. It
        /// is paid only by a failing run.
        /// </summary>
        private const int WakeDeadlineMilliseconds = 30_000;

        /// <summary>
        /// Decode workers seen executing a body, keyed by managed thread id. Static because the chunk body
        /// is a <c>delegate*</c> and captures nothing; safe because
        /// <see cref="ExclusiveProcessMeasurementCollection"/> runs one member of it at a time.
        /// </summary>
        private static readonly ConcurrentDictionary<int, Thread> _identified = new();

        /// <summary>Decode workers that executed a chunk of the wake dispatch, keyed by managed thread id.</summary>
        private static readonly ConcurrentDictionary<int, Thread> _woken = new();

        private readonly ITestOutputHelper _out;

        public DecodePoolWakeTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void ParkedWorker_IsWoken_ByTheNextDispatch()
        {
            Assert.SkipWhen(
                !OverfitParallel.DecodePoolEnabled || OverfitParallel.DecodePoolSize < 2,
                "cannot measure: there is no decode pool to wake (enabled="
                + $"{OverfitParallel.DecodePoolEnabled}, size={OverfitParallel.DecodePoolSize}). ForDecode "
                + "runs the body inline in that configuration, so there is no park and no wake.");

            var poolSize = OverfitParallel.DecodePoolSize;

            _identified.Clear();
            _woken.Clear();

            var rounds = IdentifyDecodePoolWorkers(poolSize);
            _out.WriteLine(
                $"identified {_identified.Count} of {poolSize} decode workers in {rounds} round(s)");

            Assert.SkipWhen(
                _identified.Count != poolSize,
                $"cannot measure: identified {_identified.Count} of {poolSize} decode workers over "
                + $"{rounds} dispatches. An unidentified worker cannot be shown to have parked, and one "
                + "still hot-spinning would satisfy the wake assertion without any pulse — so a subset is "
                + "neither a pass nor a failure.");

            var parkWait = Stopwatch.StartNew();
            while (!AllParked() && parkWait.ElapsedMilliseconds < ParkPollBudgetMilliseconds)
            {
                Thread.Sleep(ParkPollIntervalMilliseconds);
            }

            parkWait.Stop();
            var parked = AllParked();
            _out.WriteLine(
                $"park poll finished after {parkWait.ElapsedMilliseconds} ms, all parked = {parked}; "
                + $"states: {DescribeStates()}");

            Assert.SkipWhen(
                !parked,
                $"cannot measure: {CountParked()} of {_identified.Count} decode workers reported "
                + $"WaitSleepJoin within {ParkPollBudgetMilliseconds} ms (states: {DescribeStates()}). "
                + "Whether an idle pool parks is DecodePoolIdleBurnTests' subject; a wake test on a pool "
                + "that never parked has nothing to wake, so this is neither a pass nor a failure.");

            var parkedIds = new HashSet<int>(_identified.Keys);
            var state = new int[3];

            fixed (int* statePtr = state)
            {
                var context = new WakeContext
                {
                    Arrived = statePtr,
                    WorkerArrived = statePtr + 1,
                    Aborted = statePtr + 2,
                };

                OverfitParallel.ForDecode(0, poolSize, &WakeBody, &context);
            }

            _out.WriteLine(
                $"wake dispatch: {state[0]} bodies entered, {state[1]} of them on decode workers "
                + $"({_woken.Count} distinct), aborted = {state[2]}");

            Assert.True(
                state[2] == 0,
                $"the decode pool's wake path is broken: all {parkedIds.Count} workers were observed parked "
                + $"(WaitSleepJoin) immediately before the dispatch, and after {WakeDeadlineMilliseconds} ms "
                + "not one of them had entered a chunk body. The calling thread was blocked inside its own "
                + "chunk the whole time, so at least one chunk of this dispatch was claimable and only a "
                + "woken worker could have taken it. Check Monitor.PulseAll(_decodeParkLock) in ForDecode "
                + "(OverfitParallel.cs) and the predicate re-test in DecodeWorkerLoop.");

            Assert.True(
                state[1] > 0,
                "no chunk of the wake dispatch ran on a thread named " + DecodeThreadNamePrefix
                + "*, yet the rendezvous released. That combination should be unreachable and means this "
                + "test's own rendezvous is wrong, not that the pool is.");

            Assert.True(
                _woken.Keys.Any(parkedIds.Contains),
                "a decode worker ran a chunk, but none of the workers that ran one was among the "
                + $"{parkedIds.Count} observed parked. The wake being asserted is of a PARKED worker; a "
                + "worker that was never parked proves nothing about the pulse.");
        }

        /// <summary>
        /// Whether every identified worker currently reports <see cref="System.Threading.ThreadState.WaitSleepJoin"/>.
        /// The state is a flag set — a background worker carries <c>Background</c> too — so it is masked
        /// rather than compared.
        /// </summary>
        private static bool AllParked()
        {
            return _identified.Count > 0 && CountParked() == _identified.Count;
        }

        private static int CountParked()
        {
            var parked = 0;

            foreach (var thread in _identified.Values)
            {
                if ((thread.ThreadState & System.Threading.ThreadState.WaitSleepJoin) != 0)
                {
                    parked++;
                }
            }

            return parked;
        }

        private static string DescribeStates()
        {
            return string.Join(
                ", ",
                _identified.Values.Select(t => $"{t.Name}={t.ThreadState}"));
        }

        /// <summary>
        /// Context for the two chunk bodies. Pointers into a pinned test-owned array; nothing here outlives
        /// the <c>fixed</c> block that publishes it, which is the discipline every production caller of
        /// <see cref="OverfitParallel.ForDecode"/> follows.
        /// </summary>
        private struct WakeContext
        {
            /// <summary>Bodies entered, on any thread.</summary>
            public int* Arrived;

            /// <summary>Bodies entered on a thread the pool named <c>OverfitDecode-*</c>.</summary>
            public int* WorkerArrived;

            /// <summary>Set once a body's rendezvous deadline expired, which releases the rest immediately.</summary>
            public int* Aborted;

            /// <summary>How many chunks this dispatch published (identification only).</summary>
            public int ChunkCount;
        }

        /// <summary>
        /// Records the <see cref="Thread"/> it runs on when that thread is a decode worker, then holds until
        /// every chunk of this dispatch has arrived, so one worker cannot drain the lot. Bounded, and a
        /// short round simply identifies fewer workers.
        /// </summary>
        private static void IdentifyBody(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<WakeContext>(context);
            var thread = Thread.CurrentThread;

            if (thread.Name is not null
                && thread.Name.StartsWith(DecodeThreadNamePrefix, StringComparison.Ordinal))
            {
                _identified[thread.ManagedThreadId] = thread;
            }

            Interlocked.Increment(ref *ctx.Arrived);

            var deadline = Stopwatch.StartNew();

            while (Volatile.Read(ref *ctx.Arrived) < ctx.ChunkCount
                   && deadline.ElapsedMilliseconds < IdentificationBarrierTimeoutMilliseconds)
            {
                Thread.SpinWait(64);
            }
        }

        /// <summary>
        /// The rendezvous that makes worker participation necessary rather than likely. A body running on a
        /// decode worker records itself and returns — its arrival IS the property. A body running on the
        /// calling thread blocks until some worker has arrived, which is what stops the caller draining the
        /// dispatch by itself and leaves a chunk that only a woken worker can take.
        /// </summary>
        private static void WakeBody(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<WakeContext>(context);

            if (Volatile.Read(ref *ctx.Aborted) != 0)
            {
                return;
            }

            Interlocked.Increment(ref *ctx.Arrived);

            var thread = Thread.CurrentThread;

            if (thread.Name is not null
                && thread.Name.StartsWith(DecodeThreadNamePrefix, StringComparison.Ordinal))
            {
                _woken[thread.ManagedThreadId] = thread;
                Interlocked.Increment(ref *ctx.WorkerArrived);
                return;
            }

            var deadline = Stopwatch.StartNew();
            var spin = new SpinWait();

            while (Volatile.Read(ref *ctx.WorkerArrived) == 0)
            {
                if (deadline.ElapsedMilliseconds >= WakeDeadlineMilliseconds)
                {
                    // Set-and-return rather than throw or spin on: the remaining chunks then complete
                    // immediately, the dispatch returns normally, and the verdict is an assertion in the
                    // test rather than a hang. A hang here would take the machine-wide measurement mutex
                    // with it.
                    Volatile.Write(ref *ctx.Aborted, 1);
                    return;
                }

                spin.SpinOnce();
            }
        }

        /// <summary>
        /// Runs barrier-held dispatches until every decode worker has executed a body, or the rounds run
        /// out.
        /// </summary>
        /// <param name="poolSize">The pool's worker count, which is also the chunk count per round.</param>
        /// <returns>How many dispatches were needed.</returns>
        private static int IdentifyDecodePoolWorkers(int poolSize)
        {
            var state = new int[3];
            var rounds = 0;

            for (var round = 0; round < IdentificationRounds; round++)
            {
                rounds++;
                state.AsSpan().Clear();

                fixed (int* statePtr = state)
                {
                    var context = new WakeContext
                    {
                        Arrived = statePtr,
                        WorkerArrived = statePtr + 1,
                        Aborted = statePtr + 2,
                        ChunkCount = poolSize,
                    };

                    OverfitParallel.ForDecode(0, poolSize, &IdentifyBody, &context);
                }

                if (_identified.Count == poolSize)
                {
                    break;
                }
            }

            return rounds;
        }
    }
}
