// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Verdict for one peer group. The per-member detail lands in the caller's findings buffer; this carries
    /// the group-level answer and the numbers needed to justify it.
    /// </summary>
    /// <param name="Status">The group's state — see <see cref="DetectionStatus"/>.</param>
    /// <param name="Reason">Human-readable justification, always populated. For a non-decidable status it names
    /// what was missing, because "insufficient data" without the reason is not actionable.</param>
    /// <param name="CorrectedAlpha">The per-test significance level actually applied:
    /// <c>MaxPValue / (2 × peers)</c> — two one-sided tests per member. Surfaced so a report can show why a
    /// borderline peer was or was not flagged.</param>
    /// <param name="PeerCount">Members actually compared — the group size <b>minus</b>
    /// <paramref name="ExcludedCount"/>.</param>
    /// <param name="HighCount">Members materially above the rest.</param>
    /// <param name="LowCount">Members materially below the rest.</param>
    /// <param name="ExcludedCount">
    /// Members dropped for contributing too few usable samples — a pod that has just started, is restarting,
    /// or carries too little traffic for its latency quantiles to be defined at every scrape.
    ///
    /// <para><b>This is coverage, not noise, and it has to be surfaced.</b> A verdict computed over six of ten
    /// replicas is a weaker statement than the same verdict over all ten, and the difference is invisible in
    /// the status alone. <see cref="DetectionStatus.Healthy"/> with four members excluded is the shape that
    /// reads as health and is not.</para>
    /// </param>
    public readonly record struct PeerOutlierResult(
        DetectionStatus Status,
        string Reason,
        double CorrectedAlpha,
        int PeerCount,
        int HighCount,
        int LowCount,
        int ExcludedCount = 0)
    {
        /// <summary>Members departing from the group in either direction.</summary>
        public int OutlierCount => HighCount + LowCount;

        /// <summary>
        /// True only for <see cref="DetectionStatus.Healthy"/>. Deliberately not "not anomalous": an
        /// undecidable group is not a healthy one, and the two must not collapse at the call site.
        /// </summary>
        public bool IsHealthy => Status == DetectionStatus.Healthy;

        /// <summary>Whether every member of the group took part in the comparison.</summary>
        public bool HasFullCoverage => ExcludedCount == 0;

        /// <summary>Whether a verdict was reached at all — false for undecidable groups.</summary>
        public bool IsDecided
            => Status == DetectionStatus.Healthy || Status == DetectionStatus.Anomalous;
    }
}
