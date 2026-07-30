// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
{
    /// <summary>How <c>IncidentTracker</c> decides that two cycles are looking at the same thing.</summary>
    /// <param name="MinSubjectOverlap">
    /// Smallest share of shared <b>subjects</b> — Jaccard, intersection over union — for a group in this
    /// cycle to continue one from a previous cycle.
    ///
    /// <para><b>Subjects, not (subject, signal) pairs — and that was measured, on the lab, after the pair
    /// version failed.</b> Matching on pairs looks more precise and is brittle exactly where it matters. Once
    /// a large incident clears, what remains is one to three findings, and at that size a single signal
    /// rotating out drops the overlap below any usable threshold. On a twenty-two-cycle shadow run of one
    /// fault being introduced and removed, the pair version opened <b>eight</b> incidents instead of two,
    /// alternating opened/ongoing every other cycle. Every one of those incidents was the same pod; only the
    /// signal had changed — <c>ContainerRestarts</c>, then <c>MemoryWorkingSetBytes</c>, then
    /// <c>GcGen2HeapBytes</c>.</para>
    ///
    /// <para><b>An incident is about who is in trouble; signals are the evidence, and evidence rotates.</b>
    /// This is not a new policy either: <see cref="IncidentGroupingOptions"/> already scores
    /// <c>SamePod</c> at 1.0, so within one cycle every finding on a pod is already one incident. Keying
    /// identity on subjects is that same rule extended through time rather than a second, different one.</para>
    ///
    /// <para><b>The cost, stated plainly:</b> two genuinely independent problems on the same pod at the same
    /// time become one incident. That is already true within a cycle for the reason above, and an operator
    /// looking at that pod once — with both sets of findings in front of them — is the better failure mode
    /// than being paged twice for one machine.</para>
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
        double MinSubjectOverlap,
        int ResolveAfterMissingCycles,
        int MaxOpenIncidents)
    {
        /// <summary>
        /// Balanced: a third of the subjects shared, two missed cycles before closing, 512 tracked.
        ///
        /// <para>A third is the same bound that shows up everywhere else here. For the common case — one pod
        /// in trouble — the subject test is effectively binary, which is the stability the pair version
        /// lacked: same pod continues, different pod does not.</para>
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
            => MinSubjectOverlap > 0.0
               && MinSubjectOverlap <= 1.0
               && ResolveAfterMissingCycles >= 1
               && MaxOpenIncidents >= 1;
    }
}
