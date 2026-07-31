// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One detector's verdict about one signal on one subject over one window — the common shape that the
    /// peer-group detector, the trend detector, the hard rules and the learned scorers all reduce to, so the
    /// grouper does not need to know which of them produced what.
    /// </summary>
    /// <param name="Subject">Who this is about.</param>
    /// <param name="Signal">Metric or rule name, e.g. <c>container_cpu_cfs_throttled_periods_total</c>. Used
    /// for de-duplication, so it must be stable across pods rather than carrying pod-specific text.</param>
    /// <param name="Class">Where the signal sits between cause and consequence.</param>
    /// <param name="Start">When the anomalous behaviour began — the start of the evaluated window, not the
    /// moment of detection. Detection latency differs per detector and would corrupt the ordering.</param>
    /// <param name="End">When it was last observed.</param>
    /// <param name="Severity">Normalised 0…1. Comparable across detectors, which is the only requirement;
    /// how each detector maps its own statistic onto it is its own business.</param>
    /// <param name="Reason">Human-readable justification from the detector that produced it.</param>
    public readonly record struct SignalFinding(
        IncidentSubject Subject,
        string Signal,
        SignalClass Class,
        DateTimeOffset Start,
        DateTimeOffset End,
        double Severity,
        string Reason)
    {
        /// <summary>
        /// Optional observations behind the finding, oldest first and evenly spaced. When two findings both
        /// carry a series, the grouper may correlate them to link subjects that topology alone would leave
        /// apart. Empty by default — grouping works without it, just with less reach.
        /// </summary>
        public ReadOnlyMemory<double> Series
        {
            get; init;
        }

        /// <summary>How long the behaviour lasted.</summary>
        public TimeSpan Duration => End - Start;
    }
}
