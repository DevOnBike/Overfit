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
    public static class AllocationAssert
    {
        /// <summary>
        /// Upper bound for one-time JIT/tier-up/OSR/PGO bookkeeping charged to the measuring thread (observed ~280 B).
        /// Kept tight (1 KB) so it still catches a real per-call leak even on short measured loops: the smallest .NET
        /// heap object is ~24 B, so a genuine per-call allocation over even a 50-iteration loop is ≥ ~1200 B — above
        /// this floor — while a per-call leak over a 10k loop is hundreds of KB.
        /// </summary>
        public const long OneTimeJitNoiseFloorBytes = 1024;

        /// <summary>
        /// Asserts that <paramref name="allocatedBytes"/> measured over a hot loop reflects no per-call allocation —
        /// i.e. it is below the one-time JIT-noise floor.
        /// </summary>
        public static void NoPerCallAllocation(long allocatedBytes, string label)
        {
            Assert.True(
                allocatedBytes >= 0 && allocatedBytes < OneTimeJitNoiseFloorBytes,
                $"{label}: {allocatedBytes} B allocated in the measured loop — expected none per call " +
                $"(tolerating < {OneTimeJitNoiseFloorBytes} B one-time JIT/tier-up bookkeeping; a real per-call " +
                $"leak would be hundreds of KB).");
        }
    }
}
