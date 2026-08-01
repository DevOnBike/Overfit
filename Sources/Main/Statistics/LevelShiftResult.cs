// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>What <see cref="LevelShiftDetector"/> concluded about one series.</summary>
    /// <param name="Status">Healthy, Anomalous or InsufficientData. There is deliberately no
    /// <c>Inconclusive</c>: a two-sample comparison of two halves has no third party to be ambiguous
    /// between.</param>
    /// <param name="Direction">Which way the level moved, if it did.</param>
    /// <param name="Reason">Human-readable justification, always populated.</param>
    /// <param name="EffectSize">Cliff's delta between the halves, as a magnitude.</param>
    /// <param name="PValue">Significance of the difference.</param>
    /// <param name="Before">Median of the first half, in the signal's own units.</param>
    /// <param name="After">Median of the second half.</param>
    /// <param name="SampleCount">Usable observations across both halves.</param>
    public readonly record struct LevelShiftResult(
        DetectionStatus Status,
        TrendDirection Direction,
        string Reason,
        double EffectSize,
        double PValue,
        double Before,
        double After,
        int SampleCount)
    {
        /// <summary>The change in the signal's own units; negative when the level fell.</summary>
        public double AbsoluteChange => After - Before;

        /// <summary>
        /// The change as a fraction of the level it started from, or <see cref="double.PositiveInfinity"/>
        /// when it started from zero — which carries no proportion, and must not read as "no change".
        /// </summary>
        public double RelativeChange
            => Math.Abs(Before) <= 1e-12
                ? (Math.Abs(AbsoluteChange) <= 1e-12 ? 0.0 : double.PositiveInfinity)
                : AbsoluteChange / Math.Abs(Before);

        /// <summary>
        /// Normalised 0…1 and comparable with the other detectors' severities, which is the only requirement
        /// the incident pipeline places on it. A shift of the same size as the level it started from is taken
        /// as fully severe; the effect size scales it, so a shift the halves barely separate on cannot reach
        /// the top of the range however large it is.
        /// </summary>
        public double Severity
        {
            get
            {
                if (Status != DetectionStatus.Anomalous)
                {
                    return 0.0;
                }

                var size = double.IsFinite(RelativeChange)
                    ? Math.Clamp(Math.Abs(RelativeChange), 0.0, 1.0)
                    : 1.0;

                return Math.Clamp(size * Math.Clamp(EffectSize, 0.0, 1.0), 0.0, 1.0);
            }
        }
    }
}
