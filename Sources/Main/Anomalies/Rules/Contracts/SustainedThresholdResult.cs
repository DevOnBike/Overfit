// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Rules.Contracts
{
    /// <summary>
    /// Verdict for one signal against one threshold over one window.
    ///
    /// <para>There is deliberately no <see cref="DetectionStatus.Inconclusive"/> here. That status exists for a
    /// relative method that found contradictory evidence; an absolute threshold has no second opinion to
    /// contradict. Either enough of the window breached or it did not.</para>
    /// </summary>
    /// <param name="Status">Anomalous, Healthy, WarmingUp or InsufficientData.</param>
    /// <param name="Reason">Human-readable justification, always populated.</param>
    /// <param name="BreachFraction">Share of usable samples at or above the threshold, on 0…1.</param>
    /// <param name="BreachedSamples">How many samples breached.</param>
    /// <param name="UsableSamples">Observations the verdict rests on, after non-finite values were dropped.</param>
    /// <param name="PeakValue">Highest usable observation, or <see cref="double.NaN"/> when there were none.</param>
    /// <param name="MedianValue">Median usable observation, or <see cref="double.NaN"/> when there were none.</param>
    public readonly record struct SustainedThresholdResult(
        DetectionStatus Status,
        string Reason,
        double BreachFraction,
        int BreachedSamples,
        int UsableSamples,
        double PeakValue,
        double MedianValue)
    {
        /// <summary>True only for <see cref="DetectionStatus.Healthy"/> — an undecidable window is not a healthy one.</summary>
        public bool IsHealthy => Status == DetectionStatus.Healthy;

        /// <summary>Whether a verdict was reached at all.</summary>
        public bool IsDecided
            => Status == DetectionStatus.Healthy || Status == DetectionStatus.Anomalous;

        /// <summary>
        /// Severity on 0…1 for cross-detector ordering: <see cref="BreachFraction"/>.
        ///
        /// <para><b>Persistence, not magnitude, and that is the honest choice.</b> Crossing the threshold is
        /// what already decided the value matters — the rule's author fixed that when they set it. What remains
        /// to rank is how much of the window it held. Scaling by height instead would also be unbounded above
        /// the threshold and would need an arbitrary ceiling to fit the 0…1 contract the pipeline compares
        /// against.</para>
        /// </summary>
        public double Severity => Math.Clamp(BreachFraction, 0.0, 1.0);
    }
}
