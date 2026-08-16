// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// How the peer family decides that a deviation has stopped being news.
    ///
    /// <para><b>There is no <c>Balanced</c> here, and the omission is the design.</b> Two of these four
    /// numbers cannot be fitted by any measurement this repository can run — see
    /// <see cref="StandingReassertionInterval"/> — so a "sensible default" would be a guess wearing the
    /// clothes of a calibration. Pick <see cref="PerShift"/>, <see cref="Daily"/> or <see cref="Weekly"/>
    /// deliberately, the same way <c>PeerOutlierOptions</c> and <c>IncidentTrackingOptions</c> already
    /// require a profile to be named rather than defaulted.</para>
    ///
    /// <para><b><c>default</c> is invalid and is rejected</b> — an all-zero value would reassert continuously
    /// at zero severity, which is the worst of both behaviours.</para>
    /// </summary>
    /// <param name="StandingReassertionInterval">
    /// How long a <see cref="NoveltyKind.Standing"/> deviation stays quiet before it is reported again.
    ///
    /// <para><b>No lab spike sets this and none should be invented.</b> Unlike a detection threshold, this
    /// changes nothing about whether the deviation is correctly detected — the peer verdict is
    /// <c>Anomalous</c> every cycle either way. It governs how often an already-correct detection reminds a
    /// human, which is an operator-attention question of the kind every paging tool treats as a customer
    /// choice.</para>
    /// </param>
    /// <param name="StandingSeverityScale">
    /// What a <see cref="NoveltyKind.Standing"/> finding's severity is multiplied by, on 0…1.
    ///
    /// <para>A reasoned placeholder rather than a measurement, and lower-stakes than the interval above: it
    /// changes notification <i>priority</i>, not whether anything is reported. Half is the plain reading of
    /// "still true, but you already know".</para>
    /// </param>
    /// <param name="MinimumCycles">
    /// How many cycles of gap history are required before the change test is allowed a verdict. Below this
    /// the answer is <c>WarmingUp</c> and the finding is forwarded at full severity — insufficient history is
    /// never read as stability.
    ///
    /// <para><b>Measured only on the flat arm.</b> Replaying the recorded 15-sample <c>pj7r8</c> gap sequence
    /// (<c>MemoryWorkingSetBytes</c>, 24.4 h, gap 9.62–10.30 MB) through <c>TrendDetector</c> gave
    /// <c>Healthy</c> identically at 8, 12 and 15 — the p-value gate decides, not the sample gate, so this
    /// number is insensitive across that range for a genuinely flat series. What was <b>not</b> measured is
    /// how few cycles a real rising gap needs to be caught, so a value far below this range is not covered by
    /// evidence.</para>
    /// </param>
    /// <param name="RetainedCyclesPerSeries">
    /// How many cycles of gap history are kept per (pod, metric). The oldest is dropped once full, so this
    /// bounds both the memory and the horizon over which "changing" is judged.
    /// </param>
    public readonly record struct PeerNoveltyOptions(
        TimeSpan StandingReassertionInterval,
        double StandingSeverityScale,
        int MinimumCycles,
        int RetainedCyclesPerSeries)
    {
        /// <summary>Reasserts once per eight-hour shift, so each on-call rotation is told once.</summary>
        public static PeerNoveltyOptions PerShift => new(TimeSpan.FromHours(8), 0.5, 12, 96);

        /// <summary>Reasserts once a day.</summary>
        public static PeerNoveltyOptions Daily => new(TimeSpan.FromHours(24), 0.5, 12, 96);

        /// <summary>
        /// Reasserts once a week. The quietest setting that is still not silence — appropriate where standing
        /// differences are known, documented and genuinely uninteresting.
        /// </summary>
        public static PeerNoveltyOptions Weekly => new(TimeSpan.FromDays(7), 0.5, 12, 96);

        /// <summary>Whether these are usable at all — guards the all-zero <c>default</c> trap.</summary>
        public bool IsValid
            => StandingReassertionInterval > TimeSpan.Zero
               && StandingSeverityScale > 0.0
               && StandingSeverityScale <= 1.0
               && MinimumCycles >= 3
               && RetainedCyclesPerSeries >= MinimumCycles;
    }
}
