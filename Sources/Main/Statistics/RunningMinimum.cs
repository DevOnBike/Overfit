// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// The lowest value in a trailing window, at every position — a sliding minimum in one pass.
    ///
    /// <para><b>Why a detector needs this: some signals are sawtooth-shaped, and comparing them at an instant
    /// compares phase, not health.</b> A managed process's working set and gen-2 heap climb between collections
    /// and drop back at each one. Replicas do not collect in step — nothing synchronises their GC — so at any
    /// given scrape one replica sits near the top of its sawtooth and another near the bottom, and the two
    /// genuinely differ by hundreds of megabytes while both are perfectly healthy.</para>
    ///
    /// <para>Measured on a healthy synthetic population, that single effect produced <b>95% of all peer-group
    /// findings</b> — <c>MemoryWorkingSetBytes</c> and <c>GcGen2HeapBytes</c> together — with real median
    /// differences of 130 to 480 MB. No threshold can filter those out, because the differences are not noise:
    /// they are real, and they are meaningless.</para>
    ///
    /// <para><b>A leak moves the floor of the sawtooth; the phase does not.</b> That is the whole idea. The
    /// minimum over at least one full collection cycle is what survives every collection in that cycle, so it
    /// is phase-invariant by construction, and it is also the quantity that actually answers the operator's
    /// question — memory that a collection could not reclaim. Comparing floors across replicas is a comparison
    /// that means something; comparing instantaneous values is not.</para>
    ///
    /// <para><b>The lookback has to cover a full cycle or the fix does not work.</b> A shorter one still lands
    /// inside a single tooth and carries the phase straight through, which is why
    /// <see cref="TryFloorWindow"/> refuses to emit a window it cannot back with that much history rather than
    /// quietly returning a biased one.</para>
    ///
    /// <para>One pass, one caller-owned scratch buffer, no allocation. Every index enters and leaves the
    /// candidate deque exactly once, so the work is linear however long the lookback is — the obvious
    /// "minimum over the last k" written as a nested loop would be O(n·k). Measured against that loop:
    /// 0.32x at a lookback of 8, 0.03x at 240, flat in the lookback and zero-allocation on both sides.</para>
    ///
    /// <para><b>Measured twice against this pipeline, and it did not help either time. Nothing calls it.</b></para>
    /// <list type="bullet">
    /// <item><b>Peer comparison: a tie</b> (206/211, 181/172, 157/156 incidents a day). It could not have
    /// been anything else — the population's sawtooth amplitude is 69 MB and the absolute gate for memory is
    /// 100 MB, so the largest possible phase difference was already below the threshold.</item>
    /// <item><b>Trend detection: worse.</b> Trend findings 25→45, 27→53, 22→46, with memory going from
    /// <i>zero</i> trend findings to ten. The hypothesis had it backwards: <b>the sawtooth was protecting the
    /// trend detector, not fooling it.</b> An oscillating series has rises and falls that cancel, so
    /// Theil-Sen's median slope sits near zero and tau stays low. Taking the floor removes the oscillation
    /// and leaves long flat runs broken by a few steps in one direction — a highly monotone series, which is
    /// exactly what tau rewards.</item>
    /// </list>
    ///
    /// <para>Kept because it is correct, pinned against the naive definition, and the quantity it computes —
    /// memory a collection could not reclaim — is the right one for a leak test over a window long enough to
    /// contain several collections, which this pipeline does not currently run. <b>If that configuration
    /// never arrives, this should be deleted rather than left as something a reader assumes is in use.</b></para>
    /// </summary>
    public static class RunningMinimum
    {
        /// <summary>
        /// Longest lookback accepted. A sliding minimum is not the tool for spans this long — at some point
        /// what is wanted is a baseline over history, not a trailing extreme — and the cap keeps the scratch
        /// requirement bounded and the caller honest about that.
        /// </summary>
        public const int MaxLookbackSamples = 4096;

        /// <summary>
        /// Scratch the caller must supply for a series of this length: one index slot per sample.
        /// </summary>
        public static int RequiredScratchLength(int sourceLength)
        {
            return sourceLength < 0 ? 0 : sourceLength;
        }

        /// <summary>
        /// Writes, for every position <c>i</c>, the smallest finite value in <c>[i − lookback + 1, i]</c>.
        ///
        /// <para><b>The leading positions are deliberately included and deliberately biased.</b> Before
        /// <c>lookback − 1</c> there is not a full window to look back over, so those entries are the minimum
        /// of a shorter span and therefore still carry phase. They are emitted rather than blanked because this
        /// is the primitive, not the policy — <see cref="TryFloorWindow"/> is the entry point that enforces
        /// enough history, and it exists precisely so this trap is impossible to fall into by accident.</para>
        ///
        /// <para>Non-finite samples are skipped, not propagated: a scrape gap is missing information about
        /// memory, not a claim that memory was unmeasurable. A window with no finite sample at all yields
        /// <see cref="double.NaN"/>, which every detector downstream already reads as "no evidence".</para>
        /// </summary>
        /// <param name="source">The series, oldest first.</param>
        /// <param name="lookback">Window length in samples, 1 to <see cref="MaxLookbackSamples"/>.</param>
        /// <param name="destination">Receives one value per source sample. May alias nothing; must not overlap
        /// <paramref name="source"/>.</param>
        /// <param name="scratch">At least <see cref="RequiredScratchLength"/> entries.</param>
        public static void Compute(
            ReadOnlySpan<double> source,
            int lookback,
            Span<double> destination,
            Span<int> scratch)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(lookback, 1);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(lookback, MaxLookbackSamples);

            if (destination.Length < source.Length)
            {
                throw new ArgumentException(
                    $"Destination holds {destination.Length} of the {source.Length} values the source has.",
                    nameof(destination));
            }

            if (scratch.Length < RequiredScratchLength(source.Length))
            {
                throw new ArgumentException(
                    $"Scratch holds {scratch.Length} entries; {RequiredScratchLength(source.Length)} are "
                    + "needed, one per sample.",
                    nameof(scratch));
            }

            // Indices of the values still able to win a future window, kept in increasing order of value. A
            // value with a smaller one behind it can never be the answer again, so it is dropped on arrival of
            // that smaller one — which is what makes the whole sweep linear.
            var head = 0;
            var tail = 0;

            for (var i = 0; i < source.Length; i++)
            {
                var value = source[i];

                if (double.IsFinite(value))
                {
                    // #pragma BOUND: over the whole loop this pops at most once per index, because an index is
                    // pushed exactly once; the total is 2n, not n per position.
                    while (tail > head && source[scratch[tail - 1]] >= value)
                    {
                        tail--;
                    }

                    scratch[tail] = i;
                    tail++;
                }

                var oldest = i - lookback + 1;

                // #pragma BOUND: same accounting — each index leaves the front once and never returns.
                while (tail > head && scratch[head] < oldest)
                {
                    head++;
                }

                destination[i] = tail > head ? source[scratch[head]] : double.NaN;
            }
        }

        /// <summary>
        /// Fills <paramref name="destination"/> with the floor of the window
        /// <c>[windowStart, windowStart + windowLength)</c>, computed over <paramref name="history"/> so that
        /// <b>every</b> emitted position has a full <paramref name="lookback"/> behind it.
        ///
        /// <para>Returns <c>false</c> when the history does not reach back far enough, which is the case worth
        /// being strict about: a floor taken over less than one collection cycle is exactly the phase-dependent
        /// number this class exists to avoid, and it would look entirely reasonable in a report.</para>
        ///
        /// <para><b>The caller therefore has to fetch more than it evaluates.</b> To peer-compare a 20-minute
        /// window of a signal whose collections are 15 minutes apart, ask the metric source for 35 minutes and
        /// point <paramref name="windowStart"/> at the last 20. That is not a wart — it is the cost of the
        /// question being about a cycle rather than an instant, and the history is there to be read.</para>
        /// </summary>
        /// <param name="history">Series covering the window and the lookback that precedes it, oldest first.</param>
        /// <param name="windowStart">Index in <paramref name="history"/> where the evaluated window begins.</param>
        /// <param name="windowLength">How many samples to emit.</param>
        /// <param name="lookback">Collection cycle in samples; see <see cref="Compute"/>.</param>
        /// <param name="destination">Receives <paramref name="windowLength"/> values.</param>
        /// <param name="scratch">At least <see cref="RequiredScratchLength"/> entries for the spanned range.</param>
        public static bool TryFloorWindow(
            ReadOnlySpan<double> history,
            int windowStart,
            int windowLength,
            int lookback,
            Span<double> destination,
            Span<int> scratch)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(windowStart);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(windowLength);
            ArgumentOutOfRangeException.ThrowIfLessThan(lookback, 1);

            if (windowStart + windowLength > history.Length)
            {
                return false;
            }

            if (windowStart < lookback - 1)
            {
                return false;
            }

            if (destination.Length < windowLength)
            {
                throw new ArgumentException(
                    $"Destination holds {destination.Length} of the {windowLength} values requested.",
                    nameof(destination));
            }

            // Only the window and the lookback ahead of it can influence the answer; anything older is already
            // outside every emitted position's reach.
            var from = windowStart - lookback + 1;
            var span = history.Slice(from, windowLength + lookback - 1);

            if (scratch.Length < RequiredScratchLength(span.Length))
            {
                throw new ArgumentException(
                    $"Scratch holds {scratch.Length} entries; {RequiredScratchLength(span.Length)} are needed "
                    + "for the window plus its lookback.",
                    nameof(scratch));
            }

            // The sweep still runs over the lead-in — it has to, that is where the floor comes from — but only
            // the positions with a full lookback behind them are written out. Doing it in one pass rather than
            // calling Compute and slicing means the caller's buffer is the window it asked for, not the window
            // plus a lead-in it would then have to know to discard.
            ComputeTail(span, lookback, windowLength, destination, scratch);

            return true;
        }

        /// <summary>
        /// <see cref="Compute"/> over <paramref name="span"/>, emitting only the last
        /// <paramref name="windowLength"/> positions.
        /// </summary>
        private static void ComputeTail(
            ReadOnlySpan<double> span,
            int lookback,
            int windowLength,
            Span<double> destination,
            Span<int> scratch)
        {
            var head = 0;
            var tail = 0;
            var first = span.Length - windowLength;

            for (var i = 0; i < span.Length; i++)
            {
                var value = span[i];

                if (double.IsFinite(value))
                {
                    // #pragma BOUND: amortised — each index is pushed once and popped at most once.
                    while (tail > head && span[scratch[tail - 1]] >= value)
                    {
                        tail--;
                    }

                    scratch[tail] = i;
                    tail++;
                }

                var oldest = i - lookback + 1;

                // #pragma BOUND: each index leaves the front once.
                while (tail > head && scratch[head] < oldest)
                {
                    head++;
                }

                if (i >= first)
                {
                    destination[i - first] = tail > head ? span[scratch[head]] : double.NaN;
                }
            }
        }
    }
}
