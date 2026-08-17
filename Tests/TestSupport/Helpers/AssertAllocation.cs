// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Helpers
{
    /// <summary>
    /// Assertions for zero-allocation hot paths that are robust to JIT timing across platforms.
    ///
    /// <para>The guarantee under test is <b>zero allocation per call</b>. A genuine per-call leak allocates a whole
    /// object (≥ ~24 B) on every iteration, scaling to hundreds of KB over a 10k loop. The runtime, however, can
    /// charge a single <i>one-time</i> JIT tier-up / OSR / PGO bookkeeping allocation (observed ~280 B) to the
    /// measuring thread inside the measured window — and it lands on whichever hot-loop test the tier-up happens to
    /// occur during, which is why an exact <c>== 0</c> assertion flakes non-deterministically between Windows and
    /// Linux CI runs. This helper tolerates that one-time infrastructure blip while still catching any real per-call
    /// allocation by orders of magnitude.</para>
    /// </summary>
    public static class AssertAllocation
    {
        /// <summary>
        /// Upper bound for one-time JIT/tier-up/OSR/PGO bookkeeping charged to the measuring thread (observed ~280 B).
        /// Kept tight (1 KB) so it still catches a real per-call leak even on short measured loops: the smallest .NET
        /// heap object is ~24 B, so a genuine per-call allocation over even a 50-iteration loop is ≥ ~1200 B — above
        /// this floor — while a per-call leak over a 10k loop is hundreds of KB.
        /// </summary>
        public const long OneTimeJitNoiseFloorBytes = 1024;

        /// <summary>
        /// The rate, in bytes PER CALL, at which the quieter of the two measured halves counts as allocating
        /// on every call rather than having caught a one-time lump.
        ///
        /// <para><b>Why a rate and not a total (`XC-73`, 2026-08-17).</b>
        /// <see cref="OneTimeJitNoiseFloorBytes"/> is an ABSOLUTE budget, and it is applied to loops that
        /// differ in length by 200x — 50 calls at one site, 10 000 at another. One ambient event of ~8 KB is
        /// 0.8 B/call at the long sites and clears it untouched; at the 50-call site the same event is
        /// 163 B/call and fails the test. That is not a property of the code under measurement, it is a
        /// property of the loop length, which is why this family flaked on the short sites and not the long
        /// ones. Measured, from the failure that produced this change:
        /// <c>first half 8136 B over 25 calls · second half 0 B · gen2 +6</c>.</para>
        ///
        /// <para><b>Where 1 B/call comes from.</b> The smallest .NET heap object is ~24 B, so a genuine
        /// allocation on every call is at least 24x this floor. Keeping the floor well below 24 rather than
        /// at it is what preserves sensitivity to a leak on a path taken only sometimes: at 1 B/call this
        /// still fails a 24 B object allocated on one call in twenty-four.</para>
        /// </summary>
        public const double SustainedPerCallFloorBytes = 1.0;

        // The `NoPerCallAllocation(long allocatedBytes, string label)` overload was DELETED 2026-08-17.
        //
        // It took one total and compared it against the absolute floor, with no split and therefore no way to
        // tell a one-time lump from a per-call leak. `XC-73` migrated its last call site; leaving the method
        // behind would have left a way to reintroduce exactly the defect this class was just corrected for,
        // one call at a time and with no warning. A helper that only supports the wrong measurement is not a
        // convenience.

        /// <summary>
        /// Runs <paramref name="body"/> <paramref name="iterations"/> times and asserts it allocated nothing
        /// per call — <b>and, when it did, says enough to tell which of the three mechanisms it was without
        /// a second sighting.</b>
        ///
        /// <para><b>Why the loop lives in here rather than at the call site.</b> The overload above reports
        /// one total, which is enough to fail but not enough to diagnose: a total of, say, 4 KB is equally
        /// consistent with a one-time tier-up blip that outgrew the floor, with a warm-up that never
        /// completed, and with a genuine allocation on a rarely-taken path. Those need different fixes and
        /// the run that catches them is, by construction, the run that does not reproduce. Recorded
        /// 2026-08-14 for `XC-38`: four non-reproducing failures are on record here with no numbers between
        /// them, and the next one has to be diagnostic on its own.</para>
        ///
        /// <para><b>The discriminator is the split.</b> The loop is measured in two halves. A real per-call
        /// allocation is uniform, so both halves carry roughly the same bytes per iteration; anything
        /// one-time lands in ONE half and leaves the other clean. The GC collection counts are the third
        /// arm: a collection inside the window means another thread was allocating hard, which is the
        /// load-sensitivity these intermittent failures are suspected of. None of this costs anything on the
        /// passing path — two extra counter reads.</para>
        ///
        /// <para><b>Corrected 2026-08-17 by the first readings this thing ever produced.</b> It was written
        /// saying one-time work is <i>front-loaded</i> — first half dirty, second half clean. A loaded box
        /// gave both shapes within one run: <c>16432 B / 0 B</c> on one test and <c>0 B / 8216 B</c> on
        /// another, and the second shape was not in the reading instructions at all, so it read as
        /// unexplained. Tier-up and OSR genuinely are front-loaded; an ambient event from another thread is
        /// not, and lands wherever it happens. <b>So the half that is clean does not matter — that either
        /// half is clean is the finding</b>, and the gen counters say whether it came from outside. Both
        /// figures were ~1.6 B/call against a ~24 B floor for one object, which is by itself enough to rule
        /// a per-call leak out.</para>
        /// </summary>
        /// <param name="label">What is under measurement, quoted verbatim in the failure message.</param>
        /// <param name="iterations">Total calls; split evenly between the two measured halves.</param>
        /// <param name="body">
        /// The operation under test. Create the delegate <b>before</b> calling — the closure allocates once,
        /// at the call site, outside the measured window.
        /// </param>
        public static void NoPerCallAllocation(string label, int iterations, Action body)
        {
            var half = iterations / 2;

            var gen0 = GC.CollectionCount(0);
            var gen1 = GC.CollectionCount(1);
            var gen2 = GC.CollectionCount(2);

            var start = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < half; i++)
            {
                body();
            }

            var mid = GC.GetAllocatedBytesForCurrentThread();

            for (var i = half; i < iterations; i++)
            {
                body();
            }

            var end = GC.GetAllocatedBytesForCurrentThread();

            var total = end - start;

            if (total >= 0 && total < OneTimeJitNoiseFloorBytes)
            {
                return;
            }

            var firstHalf = mid - start;
            var secondHalf = end - mid;
            var perCall = iterations > 0 ? (double)total / iterations : 0d;
            var firstHalfPerCall = half > 0 ? (double)firstHalf / half : 0d;
            var secondHalfPerCall = iterations - half > 0 ? (double)secondHalf / (iterations - half) : 0d;

            // The verdict, in the unit the mechanism actually has (`XC-73`). A per-call allocation happens on
            // EVERY call, so it shows in BOTH halves; a one-time lump lands in one and leaves the other clean.
            // Two independent conditions, because they catch different leaks:
            //
            //   (1) SUSTAINED — even the quieter half is allocating at a real rate. This is the ordinary
            //       per-call leak, and it is length-invariant: 40 B/call fails whether the loop is 50 calls
            //       or 10 000, which the absolute budget above could not manage.
            //   (2) RARE PATH — both halves allocate something and the total is over the one-time budget. A
            //       leak taken on some calls but not all is too small per call for (1), yet it cannot be a
            //       single lump either, because a lump does not land in both halves.
            //
            // Anything else — one half genuinely clean — is one-time by construction and is NOT a failure,
            // however large. That is the whole correction: `8136 B / 0 B` over 50 calls used to be red.
            var quieterHalfPerCall = Math.Min(firstHalfPerCall, secondHalfPerCall);
            var sustained = quieterHalfPerCall >= SustainedPerCallFloorBytes;
            var rarePath = firstHalf > 0 && secondHalf > 0;

            if (!sustained && !rarePath)
            {
                // Not silent: the event is real and worth seeing, it is just not a per-call allocation.
                TestContext.Current.TestOutputHelper?.WriteLine(
                    $"{label}: {total} B in the measured window, but one half is clean "
                    + $"({firstHalf} B / {secondHalf} B over {half} / {iterations - half} calls) — a one-time "
                    + "event, not a per-call allocation. Not a failure. GC while measuring: gen0 +"
                    + $"{GC.CollectionCount(0) - gen0}, gen1 +{GC.CollectionCount(1) - gen1}, gen2 +"
                    + $"{GC.CollectionCount(2) - gen2}.");

                return;
            }

            Assert.Fail(
                (sustained
                    ? $"SUSTAINED per-call allocation: the quieter half alone is {quieterHalfPerCall:F1} B/call.\n"
                    : "RARE-PATH allocation: both halves allocated, so this is not a one-time lump.\n")
                +
                $"{label}: {total} B allocated over {iterations} calls — expected none per call.\n"
                + $"  first half:  {firstHalf} B over {half} calls "
                + $"({firstHalfPerCall:F1} B/call)\n"
                + $"  second half: {secondHalf} B over {iterations - half} calls "
                + $"({secondHalfPerCall:F1} B/call)\n"
                + $"  overall:     {perCall:F1} B/call\n"
                + $"  GC while measuring: gen0 +{GC.CollectionCount(0) - gen0}, "
                + $"gen1 +{GC.CollectionCount(1) - gen1}, gen2 +{GC.CollectionCount(2) - gen2}\n"
                + "  Reading it: EITHER half clean means the bytes were one-time and the floor, not the code, "
                + "is what to look at — front-loaded is JIT/tier-up/OSR, late is an ambient event, and the "
                + "clean half is what rules out a per-call leak either way. Both halves at a similar B/call "
                + "means a real per-call allocation; below ~24 B/call it cannot be one object per call. Gen "
                + "collections above zero mean another thread was allocating hard during the window.");
        }
    }
}
