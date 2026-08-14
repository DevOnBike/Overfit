// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// Deterministic, single-threaded tests for the decode spin-pool's claim protocol
    /// (<see cref="DecodeChunkClaim"/>). No thread, task, sleep or timing is involved: every case drives
    /// the protocol over a local word through states the dispatcher really produces.
    ///
    /// <para><b>Every state here is reached by calling the production publish routine, never by seeding a
    /// word by hand.</b> That is the whole point of <see cref="Republishing_ALargerDispatch_CannotRevive_AnExhaustedGeneration"/>:
    /// the defect it pins (`XC-50`) is a defect of <i>publication</i>, so a test that hand-writes the word
    /// it wants cannot see it — it would be asserting about a state it invented rather than about the one
    /// the dispatcher creates.</para>
    ///
    /// <para>The protocol's one <i>concurrent</i> test lives beside this file in
    /// <see cref="DecodeChunkClaimConcurrencyTests"/>, kept separate because its admissibility argument is
    /// different in kind: it is not deterministic, and what makes it safe is that every assertion holds
    /// under every legal schedule.</para>
    /// </summary>
    public sealed class DecodeChunkClaimTests
    {
        private readonly ITestOutputHelper _output;

        public DecodeChunkClaimTests(ITestOutputHelper output)
        {
            _output = output;
        }

        /// <summary>
        /// XC-50, AC1. A worker still draining generation G meets the state `ForDecode` creates when it
        /// publishes the NEXT dispatch. While the chunk count lived outside the claim word it was stored
        /// before the tag, so in that window the word carried G and its exhausted index while the bound
        /// already belonged to G+1; if G+1 had more chunks than G, the exhaustion test passed and the
        /// straggler claimed an index out of a dispatch that was not its own — executing a descriptor
        /// being overwritten and decrementing the wrong completion counter.
        ///
        /// <para>Observed before the fix, on the same scenario:
        /// <c>TryClaim returned True with index 4; word is now 0x0000000700000005</c>.</para>
        ///
        /// <para><b>What a green run here promises, and what it does not.</b> The refusal asserted in the
        /// middle of this test has <i>no independent oracle</i>. After the fix the defect state — one
        /// generation's tag beside the next generation's bound — is unrepresentable, so every mutation
        /// that reaches that assertion also reddens AC2
        /// (<see cref="Claim_YieldsExactlyChunkCountIndices_ThenRefuses"/>) or AC3
        /// (<see cref="Claim_WithASupersededGeneration_IsRefused_AndConsumesNothing"/>). Green therefore
        /// says <i>"the publication sequence produces a state the claim refuses"</i> — NOT <i>"the claim
        /// refuses for a reason nothing else covers"</i>. Nobody should read the second from it.</para>
        ///
        /// <para>The <b>final</b> block is the part this test owns alone: that publish RESETS the index
        /// rather than merely retagging the word. Nothing else in this file, and nothing else in the
        /// suite, pins it. Added 2026-08-14 after a mutation preserving the incoming index left all three
        /// tests green.</para>
        /// </summary>
        [Fact]
        public void Republishing_ALargerDispatch_CannotRevive_AnExhaustedGeneration()
        {
            const long generation = 7;
            var word = 0L;

            DecodeChunkClaim.Publish(ref word, chunkCount: 4, generation);

            // Drain G exactly the way ForDecode's caller-participation loop does. How many indices it
            // yields is AC2's subject, not this one's; what this test needs is that the dispatch was
            // really claimable and that the loop stopped because the protocol REFUSED. Without those two
            // checks the test passes vacuously the moment publish stops working — the refusal below would
            // then be a tag mismatch that proves nothing.
            var drained = 0;

            // BOUND: a correct claim yields at most chunkCount indices. The cap is not the protocol's
            // bound, it is a guard so that a mutation which never refuses fails this test instead of
            // hanging the suite.
            while (drained <= 16 && DecodeChunkClaim.TryClaim(ref word, generation, out var claimedIndex))
            {
                Assert.Equal(drained, claimedIndex);
                drained++;
            }

            Assert.True(
                drained >= 1,
                "the dispatch published for the generation under test yielded no index at all — this test "
                + "cannot say anything about a straggler until the generation it strands is real");
            Assert.True(
                drained <= 16,
                $"the claim never refused: it yielded {drained} indices for a 4-chunk dispatch and was "
                + "stopped by this test's guard cap, not by the protocol");

            // The next dispatch is LARGER than the one just drained — the ordinary case (attention's few
            // heads followed by an FFN's many outputs), and the shape that made XC-50 reachable.
            DecodeChunkClaim.Publish(ref word, chunkCount: 8, generation + 1);
            var published = word;

            var claimed = DecodeChunkClaim.TryClaim(ref word, generation, out var index);

            _output.WriteLine(
                $"after republish: word 0x{published:X16}; straggler TryClaim(gen {generation}) returned "
                + $"{claimed} with index {index}; word is now 0x{word:X16}");

            Assert.False(
                claimed,
                $"generation {generation} is exhausted at 4 chunks and must claim nothing, "
                + $"but TryClaim returned true with index {index} (word 0x{word:X16})");

            Assert.Equal(published, word);

            // The other half of what publish must do, and the half nothing else pins: it does not merely
            // retag the word, it RESETS the index. `_decodeClaim` is one static word reused by every
            // dispatch, so at republish it ALWAYS carries the previous dispatch's exhausted index — here,
            // 4. A publish that preserved it would start G+1 partway through its own range, the chunks
            // below that index would be claimed by nobody, and _decodeRemaining would never reach zero.
            // The production symptom is therefore a HANG in ForDecode's untimed completion spin, not a
            // wrong answer — which is exactly why it has to be asserted here rather than left for some
            // other test to notice.
            Assert.True(
                DecodeChunkClaim.TryClaim(ref word, generation + 1, out var firstOfNext),
                $"the generation just published (with 8 chunks) must be claimable; word 0x{word:X16}");
            Assert.Equal(0, firstOfNext);
        }

        /// <summary>
        /// XC-50, AC2. Exactly <c>chunkCount</c> claims succeed per generation, and the bound they are
        /// tested against comes from the word itself — there is no other place it could come from.
        /// </summary>
        [Fact]
        public void Claim_YieldsExactlyChunkCountIndices_ThenRefuses()
        {
            const long generation = 11;
            var word = 0L;

            DecodeChunkClaim.Publish(ref word, chunkCount: 3, generation);

            Assert.True(DecodeChunkClaim.TryClaim(ref word, generation, out var first), "claim 1 of 3");
            Assert.Equal(0, first);

            Assert.True(DecodeChunkClaim.TryClaim(ref word, generation, out var second), "claim 2 of 3");
            Assert.Equal(1, second);

            Assert.True(DecodeChunkClaim.TryClaim(ref word, generation, out var third), "claim 3 of 3");
            Assert.Equal(2, third);

            var exhausted = word;
            var fourth = DecodeChunkClaim.TryClaim(ref word, generation, out var extra);

            _output.WriteLine(
                $"4th claim on a 3-chunk dispatch returned {fourth} with index {extra}; "
                + $"word 0x{exhausted:X16} -> 0x{word:X16}");

            Assert.False(fourth, $"a 3-chunk dispatch must not yield a 4th index (got {extra})");
            Assert.Equal(0, extra);
            Assert.Equal(exhausted, word);
        }

        /// <summary>
        /// XC-50, AC3 (carried from `XC-49` while that task is open — see the XC-50 plan §9 "Should"). A
        /// claim presenting a superseded generation is refused AND consumes nothing: the word is unchanged,
        /// so the CAS never fired. An increment-then-test would burn a chunk of the new dispatch that
        /// nobody executes, and its dispatcher would wait for a completion that never arrives.
        /// </summary>
        [Fact]
        public void Claim_WithASupersededGeneration_IsRefused_AndConsumesNothing()
        {
            const long generation = 11;
            var word = 0L;

            DecodeChunkClaim.Publish(ref word, chunkCount: 4, generation + 1);
            var published = word;

            var claimed = DecodeChunkClaim.TryClaim(ref word, generation, out var index);

            _output.WriteLine(
                $"stale TryClaim(gen {generation}) against a word published for {generation + 1} returned "
                + $"{claimed} with index {index}; word 0x{published:X16} -> 0x{word:X16}");

            Assert.False(claimed, $"generation {generation} was superseded and must claim nothing");
            Assert.Equal(0, index);
            Assert.Equal(published, word);
        }

        /// <summary>
        /// XC-50 §9 (amended). A zero-chunk dispatch yields nothing. <b>`ForDecode` cannot produce this
        /// state</b> — it returns early for <c>totalWork &lt; 2</c> and for <c>_decodePoolSize &lt;= 1</c>,
        /// so <c>chunkCount</c> is at least 2 there. This pins the primitive's DEFINED BEHAVIOUR at the
        /// bottom of its range and must not be read as a production scenario: the reason it is worth
        /// pinning is that the bound is a <c>&gt;=</c> comparison, and an off-by-one there would hand out
        /// index 0 of a dispatch that has no chunk 0. Verified rather than asserted: mutating that
        /// comparison to <c>&gt;</c> reddens this test.
        ///
        /// <para><b>What it cannot distinguish, measured.</b> It does not catch a count that UNDERFLOWS
        /// past zero. A negative count's sign bits smear across the whole word, so the tag field is
        /// corrupted too and the claim is refused by the <i>generation</i> check rather than by the bound —
        /// the test goes green for a reason it does not intend. Found 2026-08-14 when a
        /// <c>chunkCount - 1</c> mutation was predicted to redden this test and did not. It is a limit of
        /// the test, not a hole in the protocol: <c>Publish</c> documents <c>[0, MaxChunkCount]</c> as the
        /// caller's obligation, and the other counts in this file catch the mutation anyway.</para>
        /// </summary>
        [Fact]
        public void Publish_WithZeroChunks_YieldsNoClaim()
        {
            const long generation = 3;
            var word = 0L;

            DecodeChunkClaim.Publish(ref word, chunkCount: 0, generation);
            var published = word;

            var claimed = DecodeChunkClaim.TryClaim(ref word, generation, out var index);

            _output.WriteLine($"0-chunk dispatch: word 0x{published:X16}; TryClaim -> {claimed}, index {index}");

            Assert.False(claimed, $"a 0-chunk dispatch yielded index {index}");
            Assert.Equal(0, index);
            Assert.Equal(published, word);
        }

        /// <summary>
        /// XC-50 §9 (amended). A one-chunk dispatch yields index 0 and then refuses. Like the zero case,
        /// <b>unreachable from `ForDecode`</b> (it runs a single-chunk range inline on the caller); it is
        /// here because 1 is the smallest count for which the bound must both admit and then refuse, which
        /// is the pair an off-by-one breaks in one direction or the other.
        /// </summary>
        [Fact]
        public void Publish_WithOneChunk_YieldsIndexZero_ThenRefuses()
        {
            const long generation = 5;
            var word = 0L;

            DecodeChunkClaim.Publish(ref word, chunkCount: 1, generation);

            Assert.True(DecodeChunkClaim.TryClaim(ref word, generation, out var only), "the single chunk");
            Assert.Equal(0, only);

            var exhausted = word;
            var second = DecodeChunkClaim.TryClaim(ref word, generation, out var extra);

            _output.WriteLine(
                $"1-chunk dispatch: 2nd claim -> {second}, index {extra}; word 0x{exhausted:X16} -> 0x{word:X16}");

            Assert.False(second, $"a 1-chunk dispatch yielded a 2nd index ({extra})");
            Assert.Equal(exhausted, word);
        }

        /// <summary>
        /// XC-50 §9 (amended) — <b>layout constraint 1, which was prose until this test existed.</b> The
        /// index occupies the LOW bits so that a claim stays a plain
        /// <c>CompareExchange(word, word + 1)</c>; that is only sound while the increment cannot carry out
        /// of the index field and into the count beside it. This drives a full
        /// <see cref="DecodeChunkClaim.MaxChunkCount"/> dispatch to exhaustion and asserts, at the top of
        /// the range, that the count field the claim is tested against is still the one publish wrote.
        ///
        /// <para>The consequence if it ever carried is not cosmetic: the bound would grow by 65536 per
        /// carry, so claims would keep succeeding past the end of the dispatch — the same shape as `XC-50`
        /// itself, reached from the other side.</para>
        ///
        /// <para>The bit-level assertions below deliberately hard-code the 32/16/16 split. A layout change
        /// must therefore be a two-place edit, which is the point: constraint 1 is a property of the split,
        /// so a test that derived the masks from production could not fail when the split moved.</para>
        /// </summary>
        [Fact]
        public void Claim_AtTheTopOfTheRange_DoesNotCarryIntoTheCountField()
        {
            const long generation = 13;
            const int count = DecodeChunkClaim.MaxChunkCount;
            var word = 0L;

            DecodeChunkClaim.Publish(ref word, count, generation);

            Assert.Equal(count, CountFieldOf(word));
            Assert.Equal(0, IndexFieldOf(word));

            // Drain the whole dispatch. Indices are checked against a counter rather than with a per-
            // iteration Assert so that a failure names the first divergence instead of drowning the run.
            var drained = 0;
            var firstWrongIndex = -1;

            // BOUND: `count` successful claims at most, and the guard cap makes a claim that never
            // refuses fail this test rather than hang the suite.
            while (drained <= count && DecodeChunkClaim.TryClaim(ref word, generation, out var claimedIndex))
            {
                if (claimedIndex != drained && firstWrongIndex < 0)
                {
                    firstWrongIndex = claimedIndex;
                }

                drained++;
            }

            _output.WriteLine(
                $"MaxChunkCount dispatch: drained {drained}; final word 0x{word:X16} "
                + $"(count field {CountFieldOf(word)}, index field {IndexFieldOf(word)})");

            Assert.Equal(-1, firstWrongIndex);
            Assert.Equal(count, drained);

            // THE ASSERTION THIS TEST EXISTS FOR. After the last claim the index field sits at its maximum
            // (count == MaxChunkCount == 0xFFFF), which is exactly the state a carry would escape from.
            Assert.Equal(count, CountFieldOf(word));
            Assert.Equal(count, IndexFieldOf(word));

            var beyond = word;
            var extra = DecodeChunkClaim.TryClaim(ref word, generation, out var extraIndex);

            Assert.False(extra, $"a {count}-chunk dispatch yielded a further index ({extraIndex})");
            Assert.Equal(beyond, word);
        }

        /// <summary>
        /// XC-50 §9 (amended). <b>This test documents a collision; it does not promise safety.</b> The tag
        /// is the low 32 bits of the generation, so two generations are indistinguishable to a claim
        /// exactly when they differ by a multiple of 2^32 — and a green run here means "the defect
        /// boundary is precisely where the comment says it is", nothing more.
        ///
        /// <para>It replaces an estimate. <c>DecodeChunkClaim</c> used to put the exposure at "days to
        /// months" of continuous decoding (and "years" before that); neither was measured, and the number
        /// a future editor actually needs is the modulus, not the calendar.</para>
        ///
        /// <para>The second half is the one that would catch a real regression: adjacent generations must
        /// stay distinguishable <i>across</i> the 32-bit boundary, where the tag runs
        /// <c>0xFFFF_FFFF</c> then <c>0</c>. A tag derived with a signed cast or a narrower mask breaks
        /// that while leaving the collision case green.</para>
        /// </summary>
        [Fact]
        public void Tag_IsTheLowThirtyTwoBitsOfGeneration_SoGenerationsTwoPow32Apart_TagCollides()
        {
            const long atBoundary = 0xFFFF_FFFFL;
            const long twoPow32Later = atBoundary + 0x1_0000_0000L;
            const long oneLater = 0x1_0000_0000L;

            var word = 0L;
            DecodeChunkClaim.Publish(ref word, chunkCount: 4, generation: twoPow32Later);

            var collided = DecodeChunkClaim.TryClaim(ref word, atBoundary, out var collidedIndex);

            _output.WriteLine(
                $"published for generation 0x{twoPow32Later:X}; a claim carrying 0x{atBoundary:X} "
                + $"returned {collided} with index {collidedIndex} — 32-bit tag collision, by design");

            Assert.True(
                collided,
                "generations 2^32 apart share a tag, so this claim IS accepted — if it were refused the "
                + "tag would not be the low 32 bits and this type's documented boundary would be wrong");
            Assert.Equal(0, collidedIndex);

            // The neighbouring case, which must NOT collide: 0xFFFF_FFFF and 0x1_0000_0000 are one apart.
            var next = 0L;
            DecodeChunkClaim.Publish(ref next, chunkCount: 4, generation: oneLater);

            var acrossTheBoundary = DecodeChunkClaim.TryClaim(ref next, atBoundary, out var strayIndex);

            Assert.False(
                acrossTheBoundary,
                $"generations 0x{atBoundary:X} and 0x{oneLater:X} are ADJACENT and must stay "
                + $"distinguishable, but a stale claim was accepted with index {strayIndex}");
            Assert.Equal(0, strayIndex);
        }

        // The claim word's field layout, mirrored here on purpose — see
        // Claim_AtTheTopOfTheRange_DoesNotCarryIntoTheCountField for why these are not read from
        // production.
        private const int CountShift = 16;
        private const long CountMask = 0xFFFFL << CountShift;
        private const long IndexMask = 0xFFFFL;

        private static int CountFieldOf(long word)
        {
            return (int)((word & CountMask) >> CountShift);
        }

        private static int IndexFieldOf(long word)
        {
            return (int)(word & IndexMask);
        }
    }
}
