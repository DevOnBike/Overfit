// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// The decode spin-pool's per-chunk work claim: one 64-bit word, advanced by a compare-and-swap.
    ///
    /// <para><b>The invariant this type exists to hold.</b> Every input to the decision "may this
    /// generation take this index" travels in the single word the CAS operates on. A claim consults no
    /// other mutable state — not a field, not a parameter fetched from somewhere the dispatcher publishes
    /// separately. That is why the chunk count rides in the word alongside the generation tag and the next
    /// index, and why <see cref="Publish"/> is the only way the word is written.</para>
    ///
    /// <para><b>Why it is its own type.</b> Living outside <see cref="OverfitParallel"/> means this code
    /// <i>cannot</i> reach that class's private dispatch state, so the invariant above is enforced by the
    /// compiler rather than by review. A future edit that wants a second location to consult does not
    /// compile.</para>
    ///
    /// <para><b>What went wrong when the count lived outside the word</b> (`XC-50`, fixed here). The
    /// dispatcher stored the NEXT dispatch's chunk count before publishing the next claim word, so for the
    /// length of that window the word carried generation <c>G</c> while the bound already belonged to
    /// <c>G+1</c>. A straggler still draining <c>G</c> passed the generation check (the word was still
    /// <c>G</c>'s) and passed the exhaustion check (<c>G</c>'s exhausted index is below <c>G+1</c>'s larger
    /// count), claimed an index out of a dispatch that was not its own, executed a descriptor being
    /// rewritten underneath it, and decremented the wrong completion counter — so a chunk of <c>G+1</c>
    /// went unexecuted and its dispatcher returned with part of the output buffer never written, silently.
    /// With the count in the word there is no third state: a <c>G</c>-tagged claim either meets <c>G</c>'s
    /// own word (tag matches, and the bound is <c>G</c>'s) or meets <c>G+1</c>'s word (tag mismatch).</para>
    /// </summary>
    internal static class DecodeChunkClaim
    {
        // Word layout, high to low: [ generation tag : 32 ][ chunk count : 16 ][ next index : 16 ].
        //
        // TWO CONSTRAINTS ARE LOAD-BEARING; the rest of the split is arbitrary.
        //
        // 1. THE INDEX OCCUPIES THE LOW BITS, so a claim stays a plain `CompareExchange(word, word + 1)`.
        //    The increment must not carry into the count field, and it cannot: the increment only happens
        //    while `next < count <= MaxChunkCount`, so the incremented index is at most MaxChunkCount and
        //    still fits its 16 bits.
        // 2. THE COUNT CAN NEVER EXCEED ITS FIELD. That is established ONCE, where the decode pool size is
        //    resolved (see OverfitParallel._decodePoolSize), not checked per dispatch — silent truncation
        //    of the bound is the failure mode if the clamp is ever dropped, and a truncated bound revives
        //    exactly the defect this layout removes.
        private const int TagShift = 32;
        private const int CountShift = 16;
        private const long CountMask = 0xFFFFL << CountShift;
        private const long IndexMask = 0xFFFFL;

        /// <summary>
        /// Largest chunk count a claim word can carry. The dispatcher's pool size is clamped to this at
        /// resolution time, which is what makes an over-wide count unrepresentable rather than something
        /// every dispatch has to test for.
        /// </summary>
        internal const int MaxChunkCount = (int)IndexMask;

        /// <summary>
        /// Publishes a dispatch onto the claim word: generation tag, chunk count and a next index of zero,
        /// in one release store. This is the <b>only</b> writer of the word besides the claim's CAS, and
        /// that is deliberate — publishing the count anywhere else is `XC-50`.
        ///
        /// <para><b>The zero index is load-bearing.</b> The caller's word is a single static reused by
        /// every dispatch, so on arrival here it always carries the <i>previous</i> dispatch's exhausted
        /// index. Preserving it would start this generation partway through its own range, leaving the
        /// chunks below that index unclaimed by anyone and the dispatcher's completion counter permanently
        /// above zero — a hang in an untimed spin, not a wrong answer. Pinned by
        /// <c>DecodeChunkClaimTests.Republishing_ALargerDispatch_CannotRevive_AnExhaustedGeneration</c>,
        /// whose closing assertions exist only for this.</para>
        /// </summary>
        /// <param name="claim">The claim word to publish onto.</param>
        /// <param name="chunkCount">Number of chunks in the dispatch. Must be in
        /// <c>[0, <see cref="MaxChunkCount"/>]</c>; the caller establishes that bound once, at pool-size
        /// resolution, so this method does not re-test it on the hot path.</param>
        /// <param name="generation">The dispatch generation; its low 32 bits become the tag.</param>
        internal static void Publish(ref long claim, int chunkCount, long generation)
        {
            Volatile.Write(ref claim, ((long)(uint)generation << TagShift) | ((long)chunkCount << CountShift));
        }

        /// <summary>
        /// Claims the next chunk of <paramref name="generation"/>, or returns <c>false</c> once that
        /// generation is drained — or has been superseded by a later dispatch.
        ///
        /// <para><b>Why one word and a CAS.</b> The generation, the bound and the next index are packed
        /// into a single 64-bit value so that "is this still my dispatch?", "is my dispatch drained?" and
        /// "take the next index" happen atomically against one another. Reading any of them separately is
        /// what allowed a descheduled worker to take an index out of the NEXT dispatch: it would run that
        /// dispatch's <c>Body</c>, decrement that dispatch's completion counter, and — because the chunk
        /// descriptors were being rewritten underneath it — could pair a <c>Body</c> from one dispatch with
        /// a <c>Context</c> from another.</para>
        ///
        /// <para>Failing the generation check must NOT consume an index, which is why this is a
        /// compare-and-swap rather than an increment followed by a test: an increment would burn a chunk
        /// of the new generation that nobody then executes, and the dispatcher would wait for a completion
        /// that never arrives. The generation check therefore stays first, before the bound.</para>
        ///
        /// <para><b>The tag is the low 32 bits of the generation, so the defect boundary is exact and is
        /// stated as arithmetic rather than as an estimate:</b> two generations are indistinguishable here
        /// precisely when they differ by a multiple of 2^32. <c>G</c> and <c>G + 2^32</c> collide — a
        /// straggler holding the first is accepted by a dispatch published for the second — while
        /// <c>G</c> and <c>G + 1</c> never do, including across the 32-bit boundary, where the tag goes
        /// <c>0xFFFF_FFFF</c> then <c>0</c>. Both directions are pinned by
        /// <c>DecodeChunkClaimTests.Tag_IsTheLowThirtyTwoBitsOfGeneration_SoGenerationsTwoPow32Apart_TagCollides</c>,
        /// which is named for the collision because a green there means "the boundary is where we said",
        /// not "this is safe". Reaching it needs 2^32 dispatches to pass while one straggler stays
        /// descheduled. Two earlier drafts of this paragraph put the time that takes at "years" and then
        /// at "days to months"; neither was measured, and neither is what a reader needs — the fact that
        /// decides whether a future edit is safe is the modulus, not the calendar.</para>
        /// </summary>
        /// <param name="claim">The dispatch's claim word, as written by <see cref="Publish"/>.</param>
        /// <param name="generation">The generation the caller believes it is draining.</param>
        /// <param name="index">The claimed chunk index; 0 when this returns <c>false</c>.</param>
        /// <returns><c>true</c> when an index was claimed for <paramref name="generation"/>.</returns>
        internal static bool TryClaim(ref long claim, long generation, out int index)
        {
            var tag = (uint)generation;

            // BOUND: retries only on a lost CAS, and every lost CAS means another thread made progress on
            // the same word — so the loop is bounded by the number of chunks in flight.
#pragma warning disable OVERFIT023
            while (true)
#pragma warning restore OVERFIT023
            {
                var current = Volatile.Read(ref claim);

                if ((uint)(current >> TagShift) != tag)
                {
                    index = 0;
                    return false;
                }

                var next = (int)(current & IndexMask);

                if (next >= (int)((current & CountMask) >> CountShift))
                {
                    index = 0;
                    return false;
                }

                // THE RETRY TAKEN WHEN THIS COMPARE-AND-SWAP LOSES — contention on this word is now
                // ARRANGED by a test, and was not when this comment was first written.
                // DecodeChunkClaimConcurrencyTests drives four threads over one word the test owns, so
                // reaching this branch no longer depends on what the suite's real ForDecode dispatches
                // happen to do on the day.
                //
                // WHAT THAT TEST DOES NOT DO IS ASSERT THE BRANCH WAS TAKEN. Whether any CAS actually
                // loses is a scheduling outcome, so an assertion on it would be a measurement of the box
                // rather than a property of the protocol — and a test whose verdict moves with load is
                // the TG-T12/TG-T13 failure mode. Its assertions (exactly `count` successes in total,
                // every index in [0, count) handed out exactly once, none outside) hold under EVERY legal
                // schedule; that invariance is the whole reason a threaded test is admissible here.
                //
                // Measured before that test existed (2026-08-14, coverlet with the repository's own
                // runsettings, three samples): the deterministic DecodeChunkClaimTests alone -> branch
                // rate 0.833 with THIS comparison the single partial branch; the full suite -> 1.000 in
                // 3 of 3. So the branch already executed — real dispatches do lose the race — but nothing
                // arranged it. A coverage figure without its arm is not evidence, and "unasserted" and
                // "uncovered" are different claims on a concurrent path.
                //
                // So read the risk before editing here, because neither the compiler nor a green suite
                // will help. The plausible regression is hoisting the Volatile.Read above the loop:
                // `current` would then never refresh, so a lost CAS would spin forever against a stale
                // expected value. That presents as a LIVELOCK — the same silent shape as a publish that
                // fails to reset the index, and the same shape ForDecode's untimed completion spin turns
                // into a hung process. The concurrency test would HANG on that mutation rather than
                // redden, which is the honest limit of what any test over this loop can promise; the
                // deadline that would convert it into a red is `XC-52`'s decision, not this type's.
                if (Interlocked.CompareExchange(ref claim, current + 1, current) == current)
                {
                    index = next;
                    return true;
                }
            }
        }
    }
}
