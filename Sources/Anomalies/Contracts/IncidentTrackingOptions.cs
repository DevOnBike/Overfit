// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// How <c>IncidentTracker</c> decides that two cycles are looking at the same thing.
    ///
    /// <para><b>Identity is the primary subject, and it is not configurable.</b> A group continues an open
    /// incident when it is centred on the same pod. There used to be a second knob here — a minimum share of
    /// shared subjects — and removing it is the point rather than an omission: an incident's periphery
    /// rotates every cycle by design, so gating identity on it produced a new incident for the same fault at
    /// an overlap of 0.33 against a bar of 0.34. <c>IncidentTracker</c> carries the run.</para>
    ///
    /// <para><b>Subjects, not (subject, signal) pairs — measured on the lab after the pair version failed.</b>
    /// Matching on pairs looks more precise and is brittle exactly where it matters. Once a large incident
    /// clears, what remains is one to three findings, and at that size a single signal rotating out drops the
    /// overlap below any usable threshold. On a twenty-two-cycle shadow run of one fault being introduced and
    /// removed, the pair version opened <b>eight</b> incidents instead of two, alternating opened/ongoing
    /// every other cycle. Every one of those incidents was the same pod; only the signal had changed —
    /// <c>ContainerRestarts</c>, then <c>MemoryWorkingSetBytes</c>, then <c>GcGen2HeapBytes</c>.</para>
    ///
    /// <para><b>An incident is about who is in trouble; signals are the evidence, and evidence rotates.</b>
    /// This is not a new policy either: <see cref="IncidentGroupingOptions"/> already scores
    /// <c>SamePod</c> at 1.0, so within one cycle every finding on a pod is already one incident. Keying
    /// identity on the subject is that same rule extended through time rather than a second, different one.
    /// </para>
    ///
    /// <para><b>The cost, stated plainly:</b> two genuinely independent problems on the same pod at the same
    /// time become one incident. That is already true within a cycle for the reason above, and an operator
    /// looking at that pod once — with both sets of findings in front of them — is the better failure mode
    /// than being paged twice for one machine.</para>
    /// </summary>
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
        int ResolveAfterMissingCycles,
        int MaxOpenIncidents)
    {
        /// <summary>
        /// Balanced: two missed cycles before closing, 512 tracked.
        ///
        /// <para><b>Use this, not <c>default</c>.</b> A <c>default(IncidentTrackingOptions)</c> is all zeros,
        /// which would close an incident the moment it flickered and track nothing; the tracker rejects it.</para>
        /// </summary>
        public static IncidentTrackingOptions Balanced => new(2, 512);

        /// <summary>Continuity favoured: slower to close, so a flickering finding cannot restart anything.</summary>
        public static IncidentTrackingOptions Sticky => new(4, 512);

        /// <summary>
        /// Separation favoured: closes on the first missed cycle.
        ///
        /// <para>Only sane where a cycle is long enough that an absence is real. At a short cadence this is
        /// the setting that manufactures resolve/open storms.</para>
        /// </summary>
        public static IncidentTrackingOptions Strict => new(1, 512);

        /// <summary>Whether these are usable at all — guards the all-zero <c>default</c> trap.</summary>
        public bool IsValid
            => ResolveAfterMissingCycles >= 1
               && MaxOpenIncidents >= 1;
    }
}
