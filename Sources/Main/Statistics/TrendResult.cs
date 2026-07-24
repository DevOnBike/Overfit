// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Verdict for one series over one window: whether it is moving, how fast, how confidently — and, when a
    /// limit was supplied, how long it has left.
    /// </summary>
    /// <param name="Status">Healthy, Anomalous, WarmingUp or InsufficientData.</param>
    /// <param name="Direction">Which way it is moving, if at all.</param>
    /// <param name="Reason">Human-readable justification, always populated.</param>
    /// <param name="SlopePerSecond">Theil-Sen slope in the series' own units per second — the median of all
    /// pairwise slopes, so a handful of spikes cannot steer it.</param>
    /// <param name="KendallTau">Monotonicity on −1…+1: the effect size. Sign matches
    /// <paramref name="Direction"/>.</param>
    /// <param name="PValue">One-sided Mann-Kendall p-value in the direction observed, corrected for
    /// autocorrelation.</param>
    /// <param name="SampleCount">Observations the verdict rests on, after filtering and any thinning.</param>
    /// <param name="Autocorrelation">Lag-1 autocorrelation of the detrended series, in 0…1 — how much the
    /// significance had to be discounted. Surfaced because a value near 1 means the window carries far less
    /// independent evidence than its sample count suggests.</param>
    /// <param name="FittedValueAtEnd">The robust fit evaluated at the last observation — the level to project
    /// from, rather than the last raw sample, which may be a spike.</param>
    /// <param name="TimeToLimit">Projected time from the last observation until the fit reaches the supplied
    /// limit. Null when no limit was given, when the series is not moving toward it, or when the answer is
    /// further away than a decade — a projection nobody should act on.</param>
    public readonly record struct TrendResult(
        DetectionStatus Status,
        TrendDirection Direction,
        string Reason,
        double SlopePerSecond,
        double KendallTau,
        double PValue,
        int SampleCount,
        double Autocorrelation,
        double FittedValueAtEnd,
        TimeSpan? TimeToLimit)
    {
        /// <summary>True only for <see cref="DetectionStatus.Healthy"/> — an undecidable series is not a
        /// healthy one.</summary>
        public bool IsHealthy => Status == DetectionStatus.Healthy;

        /// <summary>Whether a verdict was reached at all.</summary>
        public bool IsDecided
            => Status == DetectionStatus.Healthy || Status == DetectionStatus.Anomalous;

        /// <summary>Change the fit predicts over <paramref name="seconds"/>, in the series' own units.</summary>
        public double ProjectedChangeOver(double seconds) => SlopePerSecond * seconds;
    }
}
