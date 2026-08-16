// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// The one concurrent test of <see cref="DecodeChunkClaim"/>: several threads draining a single claim
    /// word that <b>this test owns</b>.
    ///
    /// <para><b>Why a threaded test is admissible here, when `TG-T12`/`TG-T13` are cautionary.</b> Those
    /// two are cautionary because they assert a quantity the <i>environment</i> produces — elapsed time,
    /// process CPU — so a loaded box moves the verdict. Threads are not the problem; environment-produced
    /// assertions are. The rule this file is written to (XC-50 plan §13.1) is: <i>a concurrency test is
    /// admissible iff every one of its assertions holds under every legal schedule.</i> Load then changes
    /// only which interleavings are sampled, never the verdict. Every assertion below — exactly
    /// <c>count</c> successful claims in total, every index in <c>[0, count)</c> handed out exactly once,
    /// none outside it — is a property of the protocol, true of any interleaving, on any number of
    /// cores.</para>
    ///
    /// <para><b>It touches no pool state.</b> <see cref="DecodeChunkClaim"/> is fieldless, lockless and
    /// static over a caller-owned <c>ref long</c>, so this drives the real protocol without going near
    /// <c>OverfitParallel</c>'s process-global dispatch state and cannot hang or perturb a concurrent
    /// <c>ForDecode</c> anywhere else in the run. That is the property the plan's original blanket refusal
    /// of a "threaded stress harness" was written before — the refusal described tests that poke the
    /// pool's own statics, which this does not.</para>
    ///
    /// <para><b>When the property is violated, is the result red or hung?</b> Both are possible and the
    /// distinction is worth stating rather than discovering. A protocol that hands out a duplicate, an
    /// out-of-range index, or too many claims goes <b>red</b> — including the case where a mutation stops
    /// the claim ever refusing, which the per-thread cap converts into a failed count instead of a
    /// runaway. A protocol that loses <i>liveness</i> — the plausible one being a
    /// <c>Volatile.Read</c> hoisted out of the CAS retry loop, which livelocks against a stale expected
    /// value — <b>hangs</b>, and no assertion here can convert that into a red: a deadline on the join
    /// would be exactly the environment-produced quantity this file exists to avoid. `--blame-hang` is the
    /// backstop for that case, and it is a backstop, not a design.</para>
    ///
    /// <para><b>What a green run does NOT say.</b> It does not say the lost-CAS retry branch executed.
    /// This test <i>arranges</i> contention; whether any compare-and-swap actually loses is a scheduling
    /// outcome, and asserting on it would be a measurement. For the same reason this test must never be a
    /// mutation's predicted victim (XC-50 plan §13.2) — a probabilistic victim makes a mutation result
    /// unreadable.</para>
    /// </summary>
    public sealed class DecodeChunkClaimConcurrencyTests
    {
        private readonly ITestOutputHelper _output;

        public DecodeChunkClaimConcurrencyTests(ITestOutputHelper output)
        {
            _output = output;
        }

        /// <summary>
        /// Exclusivity under contention: with four threads racing one published word, exactly
        /// <c>chunkCount</c> claims succeed in the whole run, every index in <c>[0, chunkCount)</c> is
        /// handed out exactly once, and no index outside that range is ever produced.
        ///
        /// <para>This is the property the compare-and-swap exists for. An increment-then-test, a
        /// non-atomic read-modify-write, or a bound re-read from anywhere but the word itself all break it
        /// under contention — though only probabilistically, which is why this test is a falsification
        /// attempt and not a coverage claim.</para>
        ///
        /// <para>The thread count and chunk count are fixed constants, deliberately: deriving either from
        /// <see cref="Environment.ProcessorCount"/> would make the amount of real parallelism — and hence
        /// what the run exercises — a property of the box. On a single-core machine every assertion below
        /// still holds; it simply samples fewer interleavings.</para>
        /// </summary>
        [Fact]
        public void Claim_UnderContention_HandsOutEveryIndexExactlyOnce_AndNoneOutsideTheRange()
        {
            const int threadCount = 4;
            const int chunkCount = 4096;
            const long generation = 23;

            // The word lives in an array element so the worker closures can take a `ref` to the same
            // storage. It is this test's own memory — no static, no pool state.
            var word = new long[1];
            DecodeChunkClaim.Publish(ref word[0], chunkCount, generation);

            var claimedPerThread = new List<int>[threadCount];
            var threads = new Thread[threadCount];

            using var start = new ManualResetEventSlim(false);

            for (var t = 0; t < threadCount; t++)
            {
                var slot = t;
                var mine = new List<int>(chunkCount);
                claimedPerThread[slot] = mine;

                threads[slot] = new Thread(() =>
                {
                    // A gate rather than a stagger: all four threads are released together so they
                    // actually contend. No timeout is involved, so nothing here is timing-dependent.
                    start.Wait();

                    // BOUND: at most chunkCount + 1 iterations per thread. The cap is NOT the protocol's
                    // bound — it is what turns "the claim never refuses" into a failed assertion on the
                    // totals below instead of a thread that never returns.
                    while (mine.Count <= chunkCount
                           && DecodeChunkClaim.TryClaim(ref word[0], generation, out var index))
                    {
                        mine.Add(index);
                    }
                })
                {
                    IsBackground = true,
                    Name = $"DecodeChunkClaimContention-{slot}",
                };

                threads[slot].Start();
            }

            start.Set();

            for (var t = 0; t < threadCount; t++)
            {
                // No timeout: a deadline would be an environment-produced quantity, which is exactly the
                // TG-T12/TG-T13 failure mode. A liveness defect therefore hangs here rather than
                // reddening — see this class's summary.
                threads[t].Join();
            }

            var seen = new bool[chunkCount];
            var duplicates = new List<int>();
            var outOfRange = new List<int>();
            var total = 0;

            for (var t = 0; t < threadCount; t++)
            {
                var mine = claimedPerThread[t];
                total += mine.Count;

                for (var i = 0; i < mine.Count; i++)
                {
                    var index = mine[i];

                    if (index < 0 || index >= chunkCount)
                    {
                        outOfRange.Add(index);
                        continue;
                    }

                    if (seen[index])
                    {
                        duplicates.Add(index);
                        continue;
                    }

                    seen[index] = true;
                }
            }

            var missing = -1;
            for (var i = 0; i < chunkCount; i++)
            {
                if (!seen[i])
                {
                    missing = i;
                    break;
                }
            }

            _output.WriteLine(
                $"{threadCount} threads over one {chunkCount}-chunk word: claims per thread "
                + $"[{string.Join(", ", claimedPerThread.Select(c => c.Count))}], total {total}; "
                + $"final word 0x{word[0]:X16}");

            Assert.True(
                outOfRange.Count == 0,
                $"{outOfRange.Count} claim(s) outside [0, {chunkCount}): "
                + string.Join(", ", outOfRange.Take(8)));
            Assert.True(
                duplicates.Count == 0,
                $"{duplicates.Count} index/indices handed out more than once: "
                + string.Join(", ", duplicates.Take(8)));
            Assert.Equal(chunkCount, total);
            Assert.True(missing < 0, $"index {missing} was never handed out to anybody");
        }
    }
}
