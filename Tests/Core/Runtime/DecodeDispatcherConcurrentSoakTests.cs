// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// `XC-52` (b) — the concurrent-caller sibling of <see cref="DecodeDispatcherInvariantTests"/>. Several
    /// test-owned threads drive <see cref="OverfitParallel.ForDecode"/> back to back, each with its own body,
    /// its own context, its own buffers and a rotating chunk count.
    ///
    /// <para><b>This is a falsification attempt and never a coverage claim.</b> A red here is a true finding
    /// — every assertion below holds under every legal schedule, so there are no false positives. A green
    /// says only that this run did not sample an interleaving that breaks the invariant, which is not
    /// evidence that no such interleaving exists. It follows that <b>this class must never be a mutation's
    /// predicted victim</b> (plan §5): a probabilistic victim makes a mutation result unreadable. Predict a
    /// victim among the sequential tests and report separately whether this one happened to notice.</para>
    ///
    /// <para><b>The invariant under test is the same one</b> (plan §1): when <c>ForDecode</c> returns, no
    /// worker will subsequently read or write that dispatch's descriptor slot, its <c>Body</c>, its
    /// <c>Context</c>, or anything the context points at. What concurrency adds is the only thing it can add
    /// — <c>_decodeGate</c> serialises the dispatches themselves, so what varies is <i>which thread
    /// publishes next</i> and how one dispatcher's stragglers line up with a different thread's publication.
    /// The sequential test can never sample that, because there is only ever one publisher.</para>
    ///
    /// <para><b>Why the per-slot in-flight counter is schedule-invariant.</b> Each thread owns a body, and a
    /// body of slot <i>s</i> is only ever reachable through a descriptor published by thread <i>s</i>.
    /// Thread <i>s</i> is inside at most one <c>ForDecode</c> at a time, so when that call returns every
    /// execution of body <i>s</i> that exists belongs to it and must be finished. A non-zero count is
    /// therefore a violation under any interleaving and on any number of cores — not a measurement of
    /// contention. Neighbouring test classes dispatch decode work concurrently and cannot move these
    /// counters: they do not run our bodies.</para>
    ///
    /// <para><b>What is asserted here is program-produced, without exception</b> — which chunk ran, against
    /// which context, how many times, and in what completion state. Nothing asserts an elapsed time, a CPU
    /// figure, a thread count, a contention rate or an ordering between independent dispatches; those are
    /// environment-produced quantities and a loaded box would move the verdict, which is the
    /// `TG-T12`/`TG-T13` failure mode. The one clock in this file is the no-progress backstop, and it is a
    /// backstop, not an assertion — see the method.</para>
    ///
    /// <para><b>One property could not be expressed and is named rather than quietly dropped.</b> "The
    /// completion counter never goes negative" is not observable from a test: <c>_decodeRemaining</c> is
    /// private with no accessor, and exposing one is a production change this task does not carry. Its
    /// observable consequence is covered indirectly and only in one direction — a counter that went negative
    /// makes <c>ForDecode</c>'s <c>!= 0</c> spin non-terminating, so the dispatching thread never returns and
    /// the backstop names it as "did not complete". <b>Measured, not asserted from the armchair</b>: under
    /// the M5 mutation (the <c>Interlocked.Decrement</c> removed from <c>ExecuteDecodeChunk</c>'s
    /// <c>finally</c>) this test failed after 60 s naming all four threads stuck on dispatch 0, with no hang
    /// dump — where `XC-52` §8 predicted M5 would hang with no victim at all. It converts the hang into a
    /// named failure for this test; it does not rescue the rest of the run, which is still holding
    /// <c>_decodeGate</c> forever.</para>
    ///
    /// <para><b>Residual risk, unchanged from the sequential test</b> (plan finding 3): detecting a torn
    /// <c>(Body, Context)</c> pair requires dereferencing a possibly-foreign pointer, so a regression can
    /// crash the run rather than redden it. The same two deliberate weakenings of realism apply — the
    /// identity is the <i>first</i> field of the context and is checked before anything inside it is
    /// dereferenced, and every context and buffer here lives on the pinned object heap for the whole test
    /// rather than in a per-dispatch <c>fixed</c> block the way production does.</para>
    /// </summary>
    public sealed class DecodeDispatcherConcurrentSoakTests
    {
        private readonly ITestOutputHelper _output;

        public DecodeDispatcherConcurrentSoakTests(ITestOutputHelper output)
        {
            _output = output;
        }

        // Fixed, not derived from Environment.ProcessorCount, for the reason DecodeChunkClaimConcurrencyTests
        // gives: deriving it would make what the run exercises a property of the box. On a small box this
        // samples fewer interleavings and every assertion still holds.
        private const int SoakThreads = 4;

        // Per-slot counters are spread a cache line apart. Not an assertion and not a measurement — it keeps
        // the cost of a dispatch dominated by the body rather than by four threads ping-ponging one line.
        private const int Stride = 16;

        // Distinct from the sequential test's stamps (52_001..52_004) on purpose: if a straggler of THAT
        // class ever wrote into a buffer of THIS one, the value it left must not read as a legal stamp here.
        private const int StampBase = 52_100;

        // The deterministic uneven-work knob. It changes WHICH interleavings are sampled and never a
        // verdict, which is what makes it admissible where a Sleep or a spin-until-a-clock would not be.
        private const int HeavyIterations = 100_000;

        // Every Nth dispatch of every thread throws from its first chunk. 7 is coprime with the twelve work
        // sizes below, so each buffer set is used in both modes over the soak rather than being pinned to one.
        private const int ThrowEvery = 7;

        // Dispatches per thread. Sized from a measurement rather than guessed: at 12 000 the whole soak ran
        // in 2 s on the dev box (32 logical, Ryzen 9 9950X3D) — 48 000 dispatches at ~42 us each, which is
        // fast-suite scale and wastes the [LongFact] budget this test is allowed to spend. 150 000 x 4
        // threads = 600 000 dispatches puts it at roughly half a minute. Nothing about the verdict depends on
        // the number; more dispatches only sample more interleavings, which is the only thing a falsification
        // attempt can buy with time.
        private const int IterationsPerThread = 150_000;

        // The backstop bounds NO PROGRESS, not the soak's duration, and the difference was measured rather
        // than reasoned. A whole-soak deadline has to be set against how long the soak legitimately takes
        // (36 s here) and therefore against how slow a loaded box can make it, which forces it so high — the
        // first version was 10 minutes — that the runner's own `--blame-hang` always fires first: under M4
        // the combined run hung for 4 minutes and reported NO test name, with this bound never reached. A
        // no-progress deadline is bounded against a single dispatch instead (~60 us), so 60 s is six orders
        // of magnitude clear of it and no amount of load can reach it, because a slow box still makes
        // progress. Its message says "did not complete", never "the protocol is violated".
        private const int NoProgressMilliseconds = 60 * 1000;

        // How long each poll of a still-running thread waits. Only the granularity of the check above.
        private const int JoinPollMilliseconds = 250;

        private const string ErrorMarker = "xc52b-body-error";

        // Small -> large -> small -> larger, repeatedly. `XC-50`'s precondition is a dispatch with MORE
        // chunks following one with fewer, and a monotone or constant sequence never creates it. chunkCount =
        // Math.Min(poolSize, work), so on a pool of 2 the counts cannot vary at all — hence the achieved
        // sequence is reported rather than assumed.
        private static readonly int[] WorkSizes = [2, 10, 3, 64, 5, 32, 2, 48, 4, 24, 2, 64];

        // Touched only by this class's own bodies. Indexed slot * Stride.
        private static readonly int[] _inFlight = new int[SoakThreads * Stride];
        private static readonly int[] _foreignContexts = new int[SoakThreads * Stride];
        private static readonly int[] _outOfRange = new int[SoakThreads * Stride];

        private static int _sink;

        /// <summary>
        /// The per-dispatch context. <c>Stamp</c> is deliberately the <b>first</b> field: a body reads it and
        /// bails before dereferencing <c>Counts</c> or <c>Values</c>, so a mismatched <c>(Body, Context)</c>
        /// pair is detected without writing through a pointer that came out of another dispatch's memory.
        ///
        /// <para><c>Throwing</c> and <c>HeavyOnFirstChunk</c> live here rather than in a static because they
        /// are per-dispatch data: the owning thread writes them before the call, and the dispatcher's
        /// publication of the generation is the release that makes them visible to a worker that acquires
        /// it. A body only reads them after the stamp has matched, so a foreign context can never be
        /// misread as an instruction.</para>
        /// </summary>
        private unsafe struct DispatchContext
        {
            public int Stamp;
            public int Length;
            public int Throwing;
            public int HeavyOnFirstChunk;
            public int* Counts;
            public int* Values;
        }

        private static unsafe void Body0(int chunkStart, int chunkEnd, void* context)
        {
            Execute(0, chunkStart, chunkEnd, context);
        }

        private static unsafe void Body1(int chunkStart, int chunkEnd, void* context)
        {
            Execute(1, chunkStart, chunkEnd, context);
        }

        private static unsafe void Body2(int chunkStart, int chunkEnd, void* context)
        {
            Execute(2, chunkStart, chunkEnd, context);
        }

        private static unsafe void Body3(int chunkStart, int chunkEnd, void* context)
        {
            Execute(3, chunkStart, chunkEnd, context);
        }

        /// <summary>
        /// The shared body. The order is the whole point: account for the body first, check the context's
        /// identity second, bounds-check every index against <i>that context's own</i> length third, and only
        /// then write.
        /// </summary>
        private static unsafe void Execute(int slot, int chunkStart, int chunkEnd, void* context)
        {
            var counter = slot * Stride;
            var ownStamp = StampBase + slot;

            // Increment BEFORE anything can return early and decrement in the finally: the counter must
            // account for every body that ran, including one that bails, because an unaccounted body is
            // exactly what this test is looking for. It decrements strictly before ExecuteDecodeChunk's own
            // decrement of _decodeRemaining, which is what makes the caller-side assertion targeted.
            Interlocked.Increment(ref _inFlight[counter]);

            try
            {
                if (context == null)
                {
                    Interlocked.Increment(ref _foreignContexts[counter]);
                    return;
                }

                ref var ctx = ref Unsafe.AsRef<DispatchContext>(context);

                // Identity before any dereference of a pointer held INSIDE the context.
                if (ctx.Stamp != ownStamp)
                {
                    Interlocked.Increment(ref _foreignContexts[counter]);
                    return;
                }

                if (ctx.Throwing != 0)
                {
                    // The throwing chunk is chunk 0, which the calling thread claims first and executes
                    // instantly; every other chunk is loaded instead. Measured under `XC-52` (a): with a body
                    // that throws everywhere, the caller drains all the chunks itself before a parked worker
                    // can wake, no worker is involved, and "the exception did not shortcut the completion
                    // wait" is untestable.
                    if (chunkStart == 0)
                    {
                        throw new InvalidOperationException(
                            $"{ErrorMarker} slot={slot} chunk=[{chunkStart},{chunkEnd})");
                    }

                    Burn(HeavyIterations);

                    return;
                }

                for (var i = chunkStart; i < chunkEnd; i++)
                {
                    if ((uint)i >= (uint)ctx.Length)
                    {
                        Interlocked.Increment(ref _outOfRange[counter]);
                        continue;
                    }

                    Interlocked.Increment(ref *(ctx.Counts + i));
                    *(ctx.Values + i) = ownStamp;
                }

                // Both shapes are sampled, alternating per dispatch, because they widen different windows and
                // the sequential test only carries the first. Heavy-on-chunk-0 loads the CALLING thread, so
                // the workers are the ones that get chunks at all; heavy-everywhere-else loads the WORKERS,
                // which is the shape that makes "the dispatcher returned while a chunk is still running"
                // observable — and it is the shape `XC-52` (a) measured as the one that catches a rethrow
                // hoisted above the completion spin.
                var heavy = ctx.HeavyOnFirstChunk != 0 ? chunkStart == 0 : chunkStart != 0;

                if (heavy)
                {
                    Burn(HeavyIterations);
                }
            }
            finally
            {
                Interlocked.Decrement(ref _inFlight[counter]);
            }
        }

        // A fixed number of arithmetic iterations, stored through a Volatile.Write so nothing folds it away.
        // Not a clock and not a sleep: no assertion depends on how long it takes.
        private static void Burn(int iterations)
        {
            var accumulator = 17;

            for (var i = 0; i < iterations; i++)
            {
                accumulator = (accumulator * 31) + i;
            }

            Volatile.Write(ref _sink, accumulator);
        }

        private static unsafe delegate*<int, int, void*, void> BodyFor(int slot)
        {
            if (slot == 0)
            {
                return &Body0;
            }

            if (slot == 1)
            {
                return &Body1;
            }

            if (slot == 2)
            {
                return &Body2;
            }

            return &Body3;
        }

        /// <summary>
        /// Four threads, each looping its own <c>ForDecode</c> dispatches over its own contexts and buffers,
        /// with a rotating work count and a throwing dispatch every seventh. Per dispatch it pins: every
        /// index of the range executed exactly once by a body carrying the dispatching thread's own stamp;
        /// nothing written outside the range; no body handed a context that is not its own; the dispatch not
        /// returning while one of its bodies is still running; and, on a throwing dispatch, the exception
        /// surfacing on the caller with no body still running and no buffer cell touched. Each buffer set is
        /// re-checked twelve dispatches later, immediately before it is reused, which is where a straggler
        /// that wrote <i>after</i> its own dispatch's check is caught.
        ///
        /// <para><b>Every wait is bounded and the bound produces a named failure.</b> Threads are background
        /// threads so a stuck one is abandoned rather than wedging the host, and the join is a poll against a
        /// <b>no-progress</b> deadline: if no thread advances its dispatch counter for a minute the test
        /// fails, naming each thread still alive and the dispatch it was on. That message means <i>did not
        /// complete — a liveness defect or a frozen box</i>, and never <i>the protocol is violated</i>. The
        /// distinction from a `TG-T12`-style assertion is the ratio, and bounding progress rather than
        /// duration is what makes the ratio large: a dispatch is ~60 µs, so a minute of no progress is six
        /// orders of magnitude clear, and a box slowed tenfold still advances. Measured, because the first
        /// version got this wrong: a whole-soak deadline had to be set against how slow a loaded box can make
        /// a 36-second soak, which pushed it to 10 minutes — and at that value the runner's own
        /// <c>--blame-hang</c> always fired first, so under M5 the run died at the runner's bound with no
        /// test name and this message was never reachable.</para>
        ///
        /// <para>The deadline lives here rather than in the dispatcher because `XC-52` (c) decided that the
        /// production completion spin gets <b>no</b> progress deadline: returning early would pop the
        /// caller's <c>fixed</c> frame while a worker may still be writing through those pointers, which is
        /// silent heap corruption in exchange for a diagnosable hang. In the test, or nowhere.</para>
        ///
        /// <para><b>It skips rather than passes when it cannot dispatch.</b> With a resolved pool size of 1
        /// or the pool switched off, <c>ForDecode</c> runs every body inline on the calling thread and every
        /// assertion below would pass with the dispatcher never running and no concurrency exercised at all —
        /// a green that says nothing. A skip is visibly not a pass.</para>
        /// </summary>
        [LongFact("36s")]
        public unsafe void ForDecode_ConcurrentCallers_FalsificationSoak_NoBodyOutlivesItsDispatch_AndNoneSeesAForeignContext()
        {
            var poolSize = OverfitParallel.DecodePoolSize;

            if (!OverfitParallel.DecodePoolEnabled)
            {
                Assert.Skip(
                    "SKIPPED, nothing was exercised: the decode spin pool is off (OVERFIT_DECODE_POOL=0, or "
                    + "the Android default), so ForDecode delegates to the capped park path and the decode "
                    + "dispatch under test never runs.");
            }

            if (poolSize <= 1)
            {
                Assert.Skip(
                    $"SKIPPED, nothing was exercised: the resolved decode pool size is {poolSize}, so "
                    + "ForDecode runs every body inline on the calling thread and publishes no dispatch. "
                    + "Every assertion in this test would pass without the dispatcher running and without "
                    + "any concurrency. It needs a pool of 2+, i.e. 3+ logical CPUs; this box reports "
                    + $"Environment.ProcessorCount = {Environment.ProcessorCount}.");
            }

            // Class-static diagnostic counters; reset so this test's verdict is about this test.
            for (var slot = 0; slot < SoakThreads; slot++)
            {
                Volatile.Write(ref _inFlight[slot * Stride], 0);
                Volatile.Write(ref _foreignContexts[slot * Stride], 0);
                Volatile.Write(ref _outOfRange[slot * Stride], 0);
            }

            var setCount = WorkSizes.Length;

            // ONE pinned lifetime for every context and buffer, rather than a fixed block per dispatch the
            // way production does it. Deliberate (see the class doc): a straggler dereferencing a stale
            // context of ours then reads memory that is still valid and leaves a detectable wrong write,
            // instead of an access violation that takes the whole run with it.
            var contexts = GC.AllocateArray<DispatchContext>(SoakThreads * setCount, pinned: true);
            var countBuffers = new int[SoakThreads * setCount][];
            var valueBuffers = new int[SoakThreads * setCount][];

            for (var slot = 0; slot < SoakThreads; slot++)
            {
                for (var set = 0; set < setCount; set++)
                {
                    var k = (slot * setCount) + set;
                    var work = WorkSizes[set];

                    countBuffers[k] = GC.AllocateArray<int>(work, pinned: true);
                    valueBuffers[k] = GC.AllocateArray<int>(work, pinned: true);

                    contexts[k] = new DispatchContext
                    {
                        Stamp = StampBase + slot,
                        Length = work,
                        Throwing = 0,
                        HeavyOnFirstChunk = 0,
                        Counts = (int*)Unsafe.AsPointer(ref countBuffers[k][0]),
                        Values = (int*)Unsafe.AsPointer(ref valueBuffers[k][0]),
                    };
                }
            }

            // A worker thread may not assert: an assertion exception escaping a Thread kills the process and
            // takes the whole run with it. Each thread records its FIRST violation here and stops; the test
            // thread reads them after the join and turns them into the failure.
            var failures = new string[SoakThreads];
            var progress = new int[SoakThreads * Stride];
            var completed = new int[SoakThreads * Stride];

            string VerifySet(int k, int slot, bool wasThrowing)
            {
                var expectedCount = wasThrowing ? 0 : 1;
                var expectedValue = wasThrowing ? 0 : StampBase + slot;
                var cells = countBuffers[k];

                for (var i = 0; i < cells.Length; i++)
                {
                    var executions = Volatile.Read(ref countBuffers[k][i]);

                    if (executions != expectedCount)
                    {
                        return $"index {i} was executed {executions} time(s), expected exactly "
                               + $"{expectedCount}"
                               + (wasThrowing ? " (a throwing dispatch writes no buffer cell)" : string.Empty);
                    }

                    var written = Volatile.Read(ref valueBuffers[k][i]);

                    if (written != expectedValue)
                    {
                        return $"index {i} carries stamp {written}, expected {expectedValue} — a body wrote "
                               + "into a context that is not its own";
                    }
                }

                return null;
            }

            void SoakBody(int slot)
            {
                var counter = slot * Stride;

                // [ThreadStatic] and default false on a fresh thread, so this cannot fire — asserted rather
                // than assumed, because if it ever did fire every dispatch below would run inline and the
                // whole soak would be green having exercised nothing.
                if (OverfitParallel.SuppressParallelismOnCurrentThread)
                {
                    failures[slot] =
                        "SuppressParallelismOnCurrentThread was set on this soak thread, which makes "
                        + "ForDecode run every body inline — nothing was exercised.";

                    return;
                }

                var used = new bool[setCount];
                var lastThrowing = new bool[setCount];

                for (var iteration = 0; iteration < IterationsPerThread; iteration++)
                {
                    Volatile.Write(ref progress[counter], iteration);

                    var set = iteration % setCount;
                    var k = (slot * setCount) + set;
                    var work = WorkSizes[set];

                    // The late-straggler sweep, and the concurrent analogue of the sequential test's final
                    // pass: this set has been untouched by us for `setCount` dispatches, so anything that
                    // moved it since its own dispatch returned outlived that dispatch.
                    if (used[set])
                    {
                        var stale = VerifySet(k, slot, lastThrowing[set]);

                        if (stale is not null)
                        {
                            failures[slot] =
                                $"slot {slot}, iteration {iteration}: buffer set {set} (work {work}) changed "
                                + $"after the dispatch that owned it returned, {setCount} dispatches ago — "
                                + stale;

                            return;
                        }
                    }

                    Array.Clear(countBuffers[k]);
                    Array.Clear(valueBuffers[k]);

                    var throwing = iteration % ThrowEvery == ThrowEvery - 1;

                    contexts[k].Throwing = throwing ? 1 : 0;
                    contexts[k].HeavyOnFirstChunk = iteration % 2 == 0 ? 1 : 0;

                    Exception caught = null;

                    try
                    {
                        OverfitParallel.ForDecode(
                            0, work, BodyFor(slot), Unsafe.AsPointer(ref contexts[k]));
                    }
                    catch (Exception ex)
                    {
                        caught = ex;
                    }

                    var chunks = Math.Min(poolSize, work);
                    var where = $"slot {slot}, iteration {iteration} (work {work}, chunks {chunks}, "
                                + $"{(throwing ? "throwing" : "normal")})";

                    // The invariant itself: nothing of this dispatch is still executing now that it returned.
                    var stillRunning = Volatile.Read(ref _inFlight[counter]);

                    if (stillRunning != 0)
                    {
                        failures[slot] =
                            $"{where}: ForDecode returned with {stillRunning} body/bodies still running — a "
                            + "body ran that was never accounted for in _decodeRemaining.";

                        return;
                    }

                    var foreign = Volatile.Read(ref _foreignContexts[counter]);

                    if (foreign != 0)
                    {
                        failures[slot] =
                            $"{where}: {foreign} body/bodies ran against a context whose stamp was not their "
                            + "own — a torn (Body, Context) pair.";

                        return;
                    }

                    var outside = Volatile.Read(ref _outOfRange[counter]);

                    if (outside != 0)
                    {
                        failures[slot] =
                            $"{where}: {outside} index/indices fell outside the context's own length — a "
                            + "chunk range that is not part of this dispatch's partition.";

                        return;
                    }

                    if (throwing && caught is null)
                    {
                        failures[slot] =
                            $"{where}: the dispatch whose first chunk throws returned normally — the "
                            + "captured exception was not rethrown to the caller.";

                        return;
                    }

                    if (throwing && !caught.Message.StartsWith(ErrorMarker, StringComparison.Ordinal))
                    {
                        failures[slot] =
                            $"{where}: the exception that surfaced was not this dispatch's own — "
                            + $"{caught.GetType().Name}: {caught.Message}";

                        return;
                    }

                    if (!throwing && caught is not null)
                    {
                        failures[slot] =
                            $"{where}: a dispatch whose bodies do not throw surfaced "
                            + $"{caught.GetType().Name}: {caught.Message} — an error captured by an earlier "
                            + "dispatch leaked into this one.";

                        return;
                    }

                    var bad = VerifySet(k, slot, throwing);

                    if (bad is not null)
                    {
                        failures[slot] = $"{where}, immediately after its dispatch returned: {bad}";

                        return;
                    }

                    used[set] = true;
                    lastThrowing[set] = throwing;
                }

                Volatile.Write(ref completed[counter], 1);
            }

            var threads = new Thread[SoakThreads];

            using (var start = new ManualResetEventSlim(false))
            {
                for (var t = 0; t < SoakThreads; t++)
                {
                    var slot = t;

                    threads[slot] = new Thread(() =>
                    {
                        // A gate rather than a stagger, so the threads actually contend for _decodeGate from
                        // the first dispatch. No timeout is involved, so nothing here is timing-dependent.
                        start.Wait();
                        SoakBody(slot);
                    })
                    {
                        // Abandonable: a thread stuck in the completion spin must fail this test rather than
                        // hold the test host open.
                        IsBackground = true,
                        Name = $"DecodeDispatcherSoak-{slot}",
                    };

                    threads[slot].Start();
                }

                start.Set();

                var stuck = new List<string>();
                var lastSeen = new int[SoakThreads];
                var lastMovement = Environment.TickCount64;

                // BOUND: every pass either joins a thread or waits JoinPollMilliseconds, and the loop leaves
                // as soon as no thread has advanced its dispatch counter for NoProgressMilliseconds — so it
                // cannot outlive the last observed advance by more than that.
                var pending = SoakThreads;

                while (pending > 0)
                {
                    pending = 0;

                    for (var t = 0; t < SoakThreads; t++)
                    {
                        if (!threads[t].Join(JoinPollMilliseconds))
                        {
                            pending++;
                        }
                    }

                    if (pending == 0)
                    {
                        break;
                    }

                    var moved = false;

                    for (var t = 0; t < SoakThreads; t++)
                    {
                        var seen = Volatile.Read(ref progress[t * Stride]);

                        if (seen != lastSeen[t])
                        {
                            lastSeen[t] = seen;
                            moved = true;
                        }
                    }

                    if (moved)
                    {
                        lastMovement = Environment.TickCount64;
                        continue;
                    }

                    if (Environment.TickCount64 - lastMovement <= NoProgressMilliseconds)
                    {
                        continue;
                    }

                    for (var t = 0; t < SoakThreads; t++)
                    {
                        if (threads[t].IsAlive)
                        {
                            stuck.Add(
                                $"{threads[t].Name} was on dispatch "
                                + $"{Volatile.Read(ref progress[t * Stride])} of {IterationsPerThread}");
                        }
                    }

                    break;
                }

                Assert.True(
                    stuck.Count == 0,
                    $"DID NOT COMPLETE: no soak thread advanced for {NoProgressMilliseconds / 1000} s. This "
                    + "says a liveness defect or a frozen box, NOT that the protocol was violated — the bound "
                    + "is on progress, six orders of magnitude above a single dispatch, so a slow box cannot "
                    + "reach it. Stuck: " + string.Join("; ", stuck));
            }

            for (var slot = 0; slot < SoakThreads; slot++)
            {
                Assert.True(failures[slot] is null, failures[slot] ?? string.Empty);
                Assert.True(
                    Volatile.Read(ref completed[slot * Stride]) == 1,
                    $"soak thread {slot} returned without completing its {IterationsPerThread} dispatches "
                    + "and without recording a violation — the loop was left by a path this test does not "
                    + "know about.");
            }

            // A body must never have seen another dispatch's context, and must never have been handed an
            // index outside its own context's length. Both are protocol violations under any schedule.
            for (var slot = 0; slot < SoakThreads; slot++)
            {
                Assert.Equal(0, Volatile.Read(ref _foreignContexts[slot * Stride]));
                Assert.Equal(0, Volatile.Read(ref _outOfRange[slot * Stride]));
                Assert.Equal(0, Volatile.Read(ref _inFlight[slot * Stride]));
            }

            var achieved = new int[setCount];
            var increases = 0;

            for (var set = 0; set < setCount; set++)
            {
                achieved[set] = Math.Min(poolSize, WorkSizes[set]);
            }

            for (var set = 1; set < setCount; set++)
            {
                if (achieved[set] > achieved[set - 1])
                {
                    increases++;
                }
            }

            var dispatches = SoakThreads * IterationsPerThread;
            var throwingDispatches = SoakThreads * (IterationsPerThread / ThrowEvery);

            _output.WriteLine(
                $"decode pool: size {poolSize}, enabled {OverfitParallel.DecodePoolEnabled}, "
                + $"ProcessorCount {Environment.ProcessorCount}");
            _output.WriteLine(
                $"{SoakThreads} concurrent callers x {IterationsPerThread} dispatches = {dispatches} "
                + $"dispatches, of which {throwingDispatches} threw from their first chunk");
            _output.WriteLine($"work counts:  [{string.Join(", ", WorkSizes)}] (cycled)");
            _output.WriteLine($"chunk counts: [{string.Join(", ", achieved)}] (min(poolSize, work))");
            _output.WriteLine(
                $"positions in the cycle whose chunk count exceeds the previous one: {increases} — this is "
                + "`XC-50`'s precondition, a larger dispatch following a smaller one.");

            if (increases == 0)
            {
                _output.WriteLine(
                    $"WARNING: with a pool of {poolSize} the chunk count never grows between dispatches, so "
                    + "this run did NOT create that precondition and proves less than it looks.");
            }

            // Deterministic given the pool size and this test's own work sizes — not an environment
            // measurement. It only fires if WorkSizes is edited into a shape that no longer creates the
            // precondition. Unreachable on a pool of 2, where every count is 2.
            if (poolSize >= 3)
            {
                Assert.True(
                    increases > 0,
                    $"the work sizes produced chunk counts [{string.Join(", ", achieved)}] on a pool of "
                    + $"{poolSize}, which never grows between dispatches — `XC-50`'s precondition is never "
                    + "created and the soak proves less than it claims.");
            }
        }
    }
}
