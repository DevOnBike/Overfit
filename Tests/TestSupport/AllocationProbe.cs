// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Measures how much a piece of code allocates, in a way that survives a busy machine.
    ///
    /// <para><b>The problem this solves is a real flake, not a tolerance dodge.</b> Several allocation tests
    /// here pass every time in isolation and fail when the suite runs on a loaded box. The cause is not
    /// measurement noise — <c>GetAllocatedBytesForCurrentThread</c> is thread-local and immune to what other
    /// threads do — it is that two things can add allocations to an otherwise allocation-free path, and both
    /// become more likely under load:</para>
    ///
    /// <list type="bullet">
    /// <item><b>The shared array pool is trimmed on gen2 collections.</b> A loaded machine collects more
    /// often, so a <c>Rent</c> that normally reuses a buffer instead allocates a fresh one — and for the
    /// pooled Mann-Whitney path that buffer is 320 KB. A test asserting <i>exactly</i> zero turns that into a
    /// guaranteed failure whenever the timing lines up.</item>
    /// <item><b>Tiered compilation promotes on a background thread.</b> Under load the promotion is late, so
    /// part of the measured loop still runs at tier 0 and the re-jit allocates on the measuring thread.</item>
    /// </list>
    ///
    /// <para><b>Both can only ever ADD.</b> Nothing about contention, collection or jitting can make a path
    /// allocate less than it really does, so the minimum across repeated attempts is the closest available
    /// estimate of the steady-state figure — and using it removes the flake without weakening the claim. A
    /// wider tolerance would have hidden a real regression of the same size; this does not.</para>
    /// </summary>
    public static class AllocationProbe
    {
        /// <summary>
        /// Runs <paramref name="action"/> <paramref name="calls"/> times and returns the total bytes
        /// allocated on this thread, taking the <b>minimum</b> across <paramref name="attempts"/> repeats.
        /// </summary>
        /// <param name="action">The code under test. Must be safe to run repeatedly.</param>
        /// <param name="calls">Calls per attempt.</param>
        /// <param name="attempts">
        /// Repeats to take the minimum of. Three is enough to step over one unlucky trim; more costs time for
        /// diminishing benefit.
        /// </param>
        public static long MinimumBytes(Action action, int calls, int attempts = 3)
        {
            ArgumentNullException.ThrowIfNull(action);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(calls);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(attempts);

            // Enough to reach tier 1 and to fill whatever pooled buffers the path rents, so the first
            // measured attempt is not paying for either.
            for (var i = 0; i < Math.Max(calls, 32); i++)
            {
                action();
            }

            var best = long.MaxValue;

            for (var attempt = 0; attempt < attempts; attempt++)
            {
                // Collected between attempts on purpose: it makes a pool trim likely to happen HERE rather
                // than inside a measurement, which is the whole point of taking a minimum afterwards.
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                // Re-warms anything the collection just took back, so the measured window is steady state.
                action();

                var before = GC.GetAllocatedBytesForCurrentThread();

                for (var i = 0; i < calls; i++)
                {
                    action();
                }

                best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
            }

            return best;
        }
    }
}
