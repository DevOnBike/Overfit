// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What the novelty gate concluded about one (pod, metric) this cycle.
    ///
    /// <para><b><see cref="Status"/> is carried separately from <see cref="Kind"/> on purpose</b>, because
    /// two very different situations both produce <see cref="NoveltyKind.New"/>: a gap that is genuinely
    /// rising (<see cref="DetectionStatus.Anomalous"/>) and a gap nobody has watched long enough to judge
    /// (<see cref="DetectionStatus.WarmingUp"/> or <see cref="DetectionStatus.InsufficientData"/>). Both are
    /// forwarded, which is right — but a gate that reported only the outcome would make "working" and "has
    /// no history yet" indistinguishable, and that is the failure mode this whole subsystem is built to
    /// avoid.</para>
    ///
    /// <para>The protocol's fifth state, <c>QueryFailed</c>, has no analogue here and never appears: this
    /// gate issues no query, it re-reads gaps the peer detector has already computed in memory. A failure to
    /// obtain the gap shows up one layer out, as the peer comparison not producing a finding at all.</para>
    /// </summary>
    /// <param name="Kind">How the finding should be classified.</param>
    /// <param name="Forward">
    /// Whether this cycle should report it. Always true for <see cref="NoveltyKind.New"/>; true for
    /// <see cref="NoveltyKind.Standing"/> only on the reassertion cadence.
    /// </param>
    /// <param name="Status">The change test's own verdict on the gap-over-time series.</param>
    /// <param name="Direction">Which way the gap is moving, when the change test decided it is.</param>
    /// <param name="Samples">Cycles of gap history the verdict rests on.</param>
    /// <param name="Reason">Why, in the change test's own words.</param>
    public readonly record struct NoveltyDecision(
        NoveltyKind Kind,
        bool Forward,
        DetectionStatus Status,
        TrendDirection Direction,
        int Samples,
        string Reason)
    {
        /// <summary>
        /// The fail-open answer: report it, classified <see cref="NoveltyKind.New"/>, because nothing is
        /// known. Used wherever the gate is switched off or cannot be consulted.
        /// </summary>
        public static NoveltyDecision Unknown
        {
            get;
        } = new(
            NoveltyKind.New,
            true,
            DetectionStatus.InsufficientData,
            TrendDirection.None,
            0,
            "No gap history for this pod and metric, so novelty is unknown and the finding is reported.");
    }
}
