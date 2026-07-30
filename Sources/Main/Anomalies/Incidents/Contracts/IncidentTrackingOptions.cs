// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
{
    /// <summary>How <c>IncidentTracker</c> decides that two cycles are looking at the same thing.</summary>
    /// <param name="MinOverlap">
    /// Smallest share of shared (subject, signal) pairs — Jaccard, intersection over union — for a group in
    /// this cycle to continue one from a previous cycle.
    ///
    /// <para><b>Overlap rather than an exact match, because the group composition moves under you.</b> A real
    /// incident gains and loses findings every cycle: a symptom crosses its threshold, a second pod joins, a
    /// noisy signal drops out. Requiring identity would open a fresh incident on every one of those, which is
    /// the behaviour this class exists to remove. Requiring too little merges unrelated problems into one
    /// ever-living incident that nobody can close.</para>
    /// </param>
    /// <param name="ResolveAfterMissingCycles">
    /// Consecutive cycles without a match before an incident closes.
    ///
    /// <para><b>One is the wrong answer, and this is a grace period rather than a delay.</b> A borderline
    /// finding sitting on its threshold drops out and returns; closing on the first miss turns that into
    /// resolve/open/resolve/open — the same alert storm in a different costume. Waiting costs a late
    /// "resolved" and buys a stable one.</para>
    /// </param>
    /// <param name="MaxOpenIncidents">
    /// Hard cap on tracked incidents. State that only ever grows is a leak in a process meant to run for
    /// months; when the cap is reached the least recently seen is dropped. Reaching it at all means a
    /// detector has stopped filtering.
    /// </param>
    public readonly record struct IncidentTrackingOptions(
        double MinOverlap,
        int ResolveAfterMissingCycles,
        int MaxOpenIncidents)
    {
        /// <summary>
        /// Balanced: a third of the pairs shared, two missed cycles before closing, 512 tracked.
        ///
        /// <para>A third is the same bound that shows up everywhere else here — below it a group has more in
        /// common with something else than with itself. Two cycles is ten minutes at the five-minute cadence
        /// the diagnostics use.</para>
        ///
        /// <para><b>Use this, not <c>default</c>.</b> A <c>default(IncidentTrackingOptions)</c> is all zeros,
        /// which would match everything to everything and never close anything; the tracker rejects it.</para>
        /// </summary>
        public static IncidentTrackingOptions Balanced => new(0.34, 2, 512);

        /// <summary>Continuity favoured: easier to keep an incident alive, slower to close it.</summary>
        public static IncidentTrackingOptions Sticky => new(0.20, 4, 512);

        /// <summary>Separation favoured: a group must look much the same to continue.</summary>
        public static IncidentTrackingOptions Strict => new(0.60, 1, 512);

        /// <summary>Whether these are usable at all — guards the all-zero <c>default</c> trap.</summary>
        public bool IsValid
            => MinOverlap > 0.0
               && MinOverlap <= 1.0
               && ResolveAfterMissingCycles >= 1
               && MaxOpenIncidents >= 1;
    }
}
