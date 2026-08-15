// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// `XC-52` (a) — a falsifier for the decode dispatcher's central invariant, which until now was a
    /// comment in <c>OverfitParallel.ForDecode</c> and nothing else.
    ///
    /// <para><b>The invariant, in the form that is observable from a client:</b> when
    /// <c>ForDecode</c> returns, no worker will subsequently read or write that dispatch's descriptor slot,
    /// its <c>Body</c>, its <c>Context</c>, or anything the context points at. That is what makes
    /// overwriting the descriptor array on the next dispatch safe, and it is what makes the API's contract
    /// — <c>context</c> is caller-owned and valid for the duration of the call — true. The task row stated
    /// it as "<c>_decodeRemaining == 0</c> implies no worker is inside <c>ExecuteDecodeChunk</c>", which is
    /// both narrower and not observable: that method decrements in its <c>finally</c>, so a worker is still
    /// inside it for a few instructions afterwards, touching nothing shared.</para>
    ///
    /// <para><b>Why the in-flight assertion is not vacuous.</b> This class's own in-flight counter is
    /// decremented at the end of the <i>body</i>, strictly before <c>ExecuteDecodeChunk</c> decrements
    /// <c>_decodeRemaining</c>. So while the invariant holds, "in-flight == 0 after <c>ForDecode</c>
    /// returns" is implied and can never fire. It can only fire when a body ran that was never accounted
    /// for in <c>_decodeRemaining</c> — a straggler executing a descriptor that is not its own, which is
    /// precisely the `XC-50` defect class.</para>
    ///
    /// <para><b>What a violation looks like here, split by which half breaks</b> (`XC-52` §3.3). A chunk
    /// executed twice, an index outside its range, or a body paired with a foreign context: <b>red</b>,
    /// deterministically. A dispatch returning while one of its bodies is still running: <b>red</b>,
    /// probabilistically — there are no false reds, but detection is not guaranteed, which is why the work
    /// is deliberately uneven (below). A chunk never claimed by anybody: <b>the run hangs</b>, inside
    /// <c>_decodeGate</c>, and no test here can convert that into a red. That is a decision, not an
    /// oversight: `XC-52` (c) rejected a progress deadline on memory safety, because returning early from
    /// the completion spin pops the caller's <c>fixed</c> frame while a worker may still be writing through
    /// those pointers — silent heap corruption instead of a diagnosable hang. The backstop is the runner's
    /// <c>--blame-hang</c> (`XC-53`), not an assertion.</para>
    ///
    /// <para><b>Residual risk that cannot be designed away</b> (`XC-52` finding 3). Detecting a torn
    /// <c>(Body, Context)</c> pair requires dereferencing a possibly-foreign pointer, so if the protocol
    /// regresses while a neighbouring test class is dispatching, that pointer may belong to another test's
    /// stack frame and the run <b>crashes</b> rather than reddens. Two things make it as survivable as it
    /// can be, and both are deliberate weakenings of realism: every body checks an identity carried in the
    /// <i>first field</i> of the context before dereferencing anything inside it, and every context and
    /// buffer here lives on the pinned object heap for the whole test rather than in a per-dispatch
    /// <c>fixed</c> block the way production does — so a straggler reading a <i>stale</i> context of ours
    /// reads memory that is still valid and produces a detectable wrong write instead of an access
    /// violation.</para>
    ///
    /// <para>No <c>Thread</c>, no <c>Task</c>, no <c>Sleep</c>, no timeout and no clock: every assertion
    /// below holds under every legal schedule, which is what makes a concurrency test admissible. Nothing
    /// here asserts an elapsed time, a core count or a contention rate — those are environment-produced
    /// quantities and a loaded box would move the verdict.</para>
    /// </summary>
    public sealed class DecodeDispatcherInvariantTests
    {
        private readonly ITestOutputHelper _output;

        public DecodeDispatcherInvariantTests(ITestOutputHelper output)
        {
            _output = output;
        }

        // Distinct per-body identities. Chosen well clear of 0 (the buffers' initial state) and of any
        // index this test uses, so a stale or unwritten cell can never be mistaken for a stamped one.
        private const int StampA = 52_001;
        private const int StampB = 52_002;
        private const int StampC = 52_003;
        private const int StampD = 52_004;

        // Uneven work: one chunk per dispatch (the one starting at the range start) does this many
        // arithmetic iterations while the others do a handful. It widens the window in which "returned
        // while a chunk is still running" is observable. It changes WHICH interleavings are sampled and
        // never a verdict, which is why it is admissible where a Sleep or a spin-until-time would not be.
        private const int HeavyIterations = 250_000;

        // The throwing dispatch loads its NON-throwing chunks instead, and more heavily, for a measured
        // reason. Its throwing chunk is chunk 0, which the calling thread claims first and executes
        // instantly, so unless the remaining chunks are slow enough to still be in flight the caller drains
        // all ten itself before a parked worker can wake — and then no worker is involved at all and there
        // is nothing for the assertions to catch. Measured on the M4 mutation (the rethrow moved above the
        // completion spin, `XC-52` §8): at 250_000 iterations and one dispatch it was caught in 2 of 3
        // runs; at this value and three dispatches, in 3 of 3. Widening the sampled interleavings is the
        // response the plan prescribes for a weak arm — the alternative, lowering the claim, is not.
        private const int ThrowingDispatchIterations = 1_000_000;

        // How many times the throwing dispatch is repeated. More samples of the same schedule-invariant
        // assertions; not a retry loop, since every repetition asserts and none is allowed to fail.
        private const int ThrowingDispatches = 3;

        // Bodies of THIS class only ever touch these. A neighbouring test class dispatching decode work
        // concurrently cannot move them.
        private static int _inFlight;
        private static int _foreignContexts;
        private static int _outOfRange;
        private static int _sink;

        /// <summary>
        /// The per-dispatch context. <c>Stamp</c> is deliberately the <b>first</b> field: a body reads it
        /// and bails before dereferencing <c>Counts</c> or <c>Values</c>, so a mismatched
        /// <c>(Body, Context)</c> pair is detected without writing through a pointer that came out of
        /// another dispatch's memory.
        /// </summary>
        private unsafe struct DispatchContext
        {
            public int Stamp;
            public int Length;
            public int* Counts;
            public int* Values;
        }

        private static unsafe void BodyA(int chunkStart, int chunkEnd, void* context)
        {
            Execute(StampA, chunkStart, chunkEnd, context);
        }

        private static unsafe void BodyB(int chunkStart, int chunkEnd, void* context)
        {
            Execute(StampB, chunkStart, chunkEnd, context);
        }

        private static unsafe void BodyC(int chunkStart, int chunkEnd, void* context)
        {
            Execute(StampC, chunkStart, chunkEnd, context);
        }

        private static unsafe void BodyD(int chunkStart, int chunkEnd, void* context)
        {
            Execute(StampD, chunkStart, chunkEnd, context);
        }

        /// <summary>
        /// The shared body. Order matters and is the whole point: account for the body first, check the
        /// context's identity second, bounds-check every index against <i>that context's own</i> length
        /// third, and only then write.
        /// </summary>
        private static unsafe void Execute(int ownStamp, int chunkStart, int chunkEnd, void* context)
        {
            // Increment BEFORE anything can return early, and decrement in the finally — the counter must
            // account for every body that ran, including one that bails, because an unaccounted body is
            // exactly what this test is looking for. It decrements strictly before ExecuteDecodeChunk's
            // own decrement of _decodeRemaining, which is what makes the caller-side assertion targeted.
            Interlocked.Increment(ref _inFlight);

            try
            {
                if (context == null)
                {
                    Interlocked.Increment(ref _foreignContexts);
                    return;
                }

                ref var ctx = ref Unsafe.AsRef<DispatchContext>(context);

                // Identity before any dereference of a pointer held INSIDE the context (finding 3).
                if (ctx.Stamp != ownStamp)
                {
                    Interlocked.Increment(ref _foreignContexts);
                    return;
                }

                for (var i = chunkStart; i < chunkEnd; i++)
                {
                    if ((uint)i >= (uint)ctx.Length)
                    {
                        Interlocked.Increment(ref _outOfRange);
                        continue;
                    }

                    Interlocked.Increment(ref *(ctx.Counts + i));
                    *(ctx.Values + i) = ownStamp;
                }

                if (chunkStart == 0)
                {
                    Burn(HeavyIterations);
                }
            }
            finally
            {
                Interlocked.Decrement(ref _inFlight);
            }
        }

        /// <summary>
        /// Throws on the chunk that starts at the range start and does the heavy work on every other
        /// chunk. The asymmetry is what gives the throwing dispatch any falsification power: with a body
        /// that throws everywhere, every chunk finishes immediately and "the exception did not shortcut the
        /// completion wait" is untestable.
        /// </summary>
        private static unsafe void ThrowOnFirstChunkBody(int chunkStart, int chunkEnd, void* context)
        {
            Interlocked.Increment(ref _inFlight);

            try
            {
                if (chunkStart == 0)
                {
                    throw new InvalidOperationException($"xc52-body-error chunk=[{chunkStart},{chunkEnd})");
                }

                Burn(ThrowingDispatchIterations);
            }
            finally
            {
                Interlocked.Decrement(ref _inFlight);
            }
        }

        // A fixed number of arithmetic iterations, stored through a Volatile.Write so nothing folds it
        // away. Not a clock and not a sleep: no assertion depends on how long it takes.
        private static void Burn(int iterations)
        {
            var accumulator = 17;

            for (var i = 0; i < iterations; i++)
            {
                accumulator = (accumulator * 31) + i;
            }

            Volatile.Write(ref _sink, accumulator);
        }

        private static unsafe delegate*<int, int, void*, void> BodyFor(int round)
        {
            var slot = round % 4;

            if (slot == 0)
            {
                return &BodyA;
            }

            if (slot == 1)
            {
                return &BodyB;
            }

            if (slot == 2)
            {
                return &BodyC;
            }

            return &BodyD;
        }

        private static int StampFor(int round)
        {
            var slot = round % 4;

            if (slot == 0)
            {
                return StampA;
            }

            if (slot == 1)
            {
                return StampB;
            }

            if (slot == 2)
            {
                return StampC;
            }

            return StampD;
        }

        /// <summary>
        /// Back-to-back <c>ForDecode</c> dispatches on the calling thread, each with its own body, its own
        /// context and its own chunk count. Pins, per dispatch: every index in the range is executed
        /// exactly once; every value written carries the dispatching body's own stamp; nothing outside the
        /// range is written; no body ever sees a context that is not its own; and the dispatch does not
        /// return while one of its bodies is still running. A final sweep re-checks every buffer after all
        /// dispatches, which is where a straggler that wrote <i>after</i> its dispatch's own check is
        /// caught. Then the exception path, which nothing covered: a dispatch whose first chunk throws must
        /// surface that exception on the caller <i>and</i> must not return while another of its bodies is
        /// still running, because <c>ExecuteDecodeChunk</c> captures into <c>Error</c> and still decrements
        /// in its <c>finally</c>, and the completion spin precedes the rethrow.
        ///
        /// <para><b>The work counts alternate small → large repeatedly on purpose.</b> `XC-50`'s
        /// precondition is a dispatch with MORE chunks following one with fewer — a monotone or constant
        /// sequence never creates it. <c>chunkCount = Math.Min(poolSize, totalWork)</c>, so on a pool of 2
        /// the counts cannot vary at all and the run proves less than it looks; the achieved sequence is
        /// therefore reported, not assumed.</para>
        ///
        /// <para><b>It skips rather than passes when it cannot dispatch.</b> With a resolved pool size of 1
        /// (a 2-vCPU box) or the pool switched off, <c>ForDecode</c> runs every body inline and every
        /// assertion below would pass without the dispatcher ever running — a green that says nothing. A
        /// skip is visibly not a pass; failing instead would be a false red on a legitimate small runner.
        /// </para>
        /// </summary>
        [Fact]
        public unsafe void ForDecode_BackToBackDispatches_NoBodyOutlivesItsDispatch_AndNoneSeesAForeignContext()
        {
            var poolSize = OverfitParallel.DecodePoolSize;

            if (!OverfitParallel.DecodePoolEnabled)
            {
                Assert.Skip(
                    "SKIPPED, nothing was exercised: the decode spin pool is off (OVERFIT_DECODE_POOL=0, or "
                    + "the Android default), so ForDecode delegates to the capped park path and the decode "
                    + "claim protocol under test never runs.");
            }

            if (poolSize <= 1)
            {
                Assert.Skip(
                    $"SKIPPED, nothing was exercised: the resolved decode pool size is {poolSize}, so "
                    + "ForDecode runs every body inline and publishes no dispatch. Every assertion in this "
                    + "test would pass without the dispatcher running. It needs a pool of 2+, i.e. 3+ "
                    + $"logical CPUs; this box reports Environment.ProcessorCount = {Environment.ProcessorCount}.");
            }

            if (OverfitParallel.SuppressParallelismOnCurrentThread)
            {
                Assert.Skip(
                    "SKIPPED, nothing was exercised: SuppressParallelismOnCurrentThread is set on this "
                    + "thread (leaked by an earlier test on the same thread), which makes ForDecode run "
                    + "every body inline.");
            }

            // Diagnostic counters are class-static; reset them so this test's verdict is about this test.
            Volatile.Write(ref _inFlight, 0);
            Volatile.Write(ref _foreignContexts, 0);
            Volatile.Write(ref _outOfRange, 0);

            // Small → large → small → larger …; the last entry is the recovery dispatch run after the
            // throwing one.
            int[] works = [2, 10, 3, 64, 5, 32, 2, 48, 4, 24, 2, 64, 16];
            var rounds = works.Length - 1;

            // ONE pinned lifetime for every context and buffer, rather than a fixed block per dispatch the
            // way production does it. Deliberate (see the class doc): a straggler dereferencing a stale
            // context of ours then reads memory that is still valid and leaves a detectable wrong write,
            // instead of an access violation that takes the whole run with it.
            var contexts = GC.AllocateArray<DispatchContext>(works.Length, pinned: true);
            var countBuffers = new int[works.Length][];
            var valueBuffers = new int[works.Length][];

            for (var r = 0; r < works.Length; r++)
            {
                countBuffers[r] = GC.AllocateArray<int>(works[r], pinned: true);
                valueBuffers[r] = GC.AllocateArray<int>(works[r], pinned: true);

                contexts[r] = new DispatchContext
                {
                    Stamp = StampFor(r),
                    Length = works[r],
                    Counts = (int*)Unsafe.AsPointer(ref countBuffers[r][0]),
                    Values = (int*)Unsafe.AsPointer(ref valueBuffers[r][0]),
                };
            }

            var achieved = new int[works.Length];

            void RunAndCheckRound(int round)
            {
                var work = works[round];
                achieved[round] = Math.Min(poolSize, work);

                OverfitParallel.ForDecode(
                    0, work, BodyFor(round), Unsafe.AsPointer(ref contexts[round]));

                // The invariant itself: nothing of this dispatch is still executing now that it returned.
                var stillRunning = Volatile.Read(ref _inFlight);
                Assert.True(
                    stillRunning == 0,
                    $"round {round} (work {work}, chunks {achieved[round]}): ForDecode returned with "
                    + $"{stillRunning} body/bodies still running — a body ran that was never accounted for "
                    + "in _decodeRemaining.");

                // Checked per round rather than once at the end so the round is named in the failure, and
                // checked BEFORE the buffers so that when a body is handed a context that is not its own,
                // the test says so instead of reporting the missing writes that follow from it. Both are
                // protocol violations under any schedule.
                var foreign = Volatile.Read(ref _foreignContexts);
                Assert.True(
                    foreign == 0,
                    $"round {round} (work {work}, chunks {achieved[round]}): {foreign} body/bodies ran "
                    + "against a context whose stamp was not their own — a torn (Body, Context) pair.");

                var outside = Volatile.Read(ref _outOfRange);
                Assert.True(
                    outside == 0,
                    $"round {round} (work {work}, chunks {achieved[round]}): {outside} index/indices fell "
                    + "outside the context's own length — a chunk range that is not part of this "
                    + "dispatch's partition.");

                CheckBuffers(round, "immediately after its dispatch returned");
            }

            void CheckBuffers(int round, string when)
            {
                var work = works[round];
                var stamp = StampFor(round);
                var badCount = -1;
                var badCountValue = 0;
                var badStamp = -1;
                var badStampValue = 0;

                for (var i = 0; i < work; i++)
                {
                    var executions = Volatile.Read(ref countBuffers[round][i]);

                    if (badCount < 0 && executions != 1)
                    {
                        badCount = i;
                        badCountValue = executions;
                    }

                    var written = Volatile.Read(ref valueBuffers[round][i]);

                    if (badStamp < 0 && written != stamp)
                    {
                        badStamp = i;
                        badStampValue = written;
                    }
                }

                Assert.True(
                    badCount < 0,
                    $"round {round} (work {work}, chunks {achieved[round]}), {when}: index {badCount} was "
                    + $"executed {badCountValue} times, expected exactly 1.");

                Assert.True(
                    badStamp < 0,
                    $"round {round} (work {work}, chunks {achieved[round]}), {when}: index {badStamp} "
                    + $"carries stamp {badStampValue}, expected this dispatch's own stamp {stamp} — a body "
                    + "wrote into a context that is not its own.");
            }

            for (var r = 0; r < rounds; r++)
            {
                RunAndCheckRound(r);
            }

            // A body must never have seen another dispatch's context, and must never have been handed an
            // index outside its own context's length. Both are protocol violations under any schedule.
            Assert.Equal(0, Volatile.Read(ref _foreignContexts));
            Assert.Equal(0, Volatile.Read(ref _outOfRange));

            // Second sweep over everything: a straggler that wrote AFTER its own round's check is only
            // visible here.
            for (var r = 0; r < rounds; r++)
            {
                CheckBuffers(r, "on the final sweep, after every dispatch had returned");
            }

            // The exception path is part of the same invariant and nothing covered it: ExecuteDecodeChunk
            // captures into Error and still decrements in its finally, and the completion spin precedes the
            // rethrow — so a throwing dispatch must ALSO surface its exception and must ALSO not return
            // while a body is running.
            for (var attempt = 0; attempt < ThrowingDispatches; attempt++)
            {
                var thrown = Assert.Throws<InvalidOperationException>(() =>
                {
                    OverfitParallel.ForDecode(0, 32, &ThrowOnFirstChunkBody, null);
                });

                Assert.StartsWith("xc52-body-error", thrown.Message);

                var runningAtThrow = Volatile.Read(ref _inFlight);
                Assert.True(
                    runningAtThrow == 0,
                    $"throwing dispatch {attempt}: the exception surfaced with {runningAtThrow} body/bodies "
                    + "still running — the rethrow shortcut the completion wait.");
            }

            // …and the pool is clean afterwards: the captured Error must not leak into the next dispatch.
            RunAndCheckRound(rounds);

            Assert.Equal(0, Volatile.Read(ref _foreignContexts));
            Assert.Equal(0, Volatile.Read(ref _outOfRange));

            var increases = 0;

            for (var r = 1; r <= rounds; r++)
            {
                if (achieved[r] > achieved[r - 1])
                {
                    increases++;
                }
            }

            _output.WriteLine(
                $"decode pool: size {poolSize}, enabled {OverfitParallel.DecodePoolEnabled}, "
                + $"ProcessorCount {Environment.ProcessorCount}");
            _output.WriteLine($"work counts:  [{string.Join(", ", works)}]");
            _output.WriteLine($"chunk counts: [{string.Join(", ", achieved)}] (min(poolSize, work))");
            _output.WriteLine(
                $"dispatches whose chunk count exceeded the previous one: {increases} — this is `XC-50`'s "
                + "precondition, a larger dispatch following a smaller one.");

            if (increases == 0)
            {
                _output.WriteLine(
                    $"WARNING: with a pool of {poolSize} the chunk count never grows between dispatches, so "
                    + "this run did NOT create that precondition and proves less than it looks.");
            }

            // Deterministic given the pool size and this test's own work sequence — not an environment
            // measurement. It only fires if the sequence above is edited into a shape that no longer
            // creates the precondition. Unreachable on a pool of 2, where every count is 2.
            if (poolSize >= 3)
            {
                Assert.True(
                    increases > 0,
                    $"the work sequence produced chunk counts [{string.Join(", ", achieved)}] on a pool of "
                    + $"{poolSize}, which never grows between dispatches — `XC-50`'s precondition is never "
                    + "created and the test proves less than it claims.");
            }
        }
    }
}
