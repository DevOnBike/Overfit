// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// The size gate when the series has no median to be relative to.
    ///
    /// <para>Found on the cluster lab: <c>GcPauseRatio</c> is identically zero on healthy replicas apart from
    /// occasional readings around 10⁻⁵, and the detector raised a severity-0.59 incident on it. The rule had
    /// been <c>material = !scaleIsUsable || …</c> — when the size could not be judged, it counted as large.
    /// The reason string said <i>"a measurable amount (no usable scale: the series sits at zero)"</i> and the
    /// verdict was <c>Anomalous</c> anyway.</para>
    ///
    /// <para><b>Both directions are pinned here, because this is easy to over-correct.</b> Silencing every
    /// zero-median series would throw away the case that matters most — a signal climbing away from zero,
    /// which is what an error rate leaving zero or a queue starting to build looks like, and whose median is
    /// zero for most of the window in which it is worth catching.</para>
    /// </summary>
    public sealed class TrendDetectorZeroScaleTests
    {
        // 120 samples, matching the existing zero-series fixture. Sixty is not enough: consecutive scrapes of
        // a ramp are almost perfectly autocorrelated (lag-1 0.95), the AR(1) correction inflates the variance
        // accordingly, and a genuine climb lands at p = 0.073 — rejected for the right reason, but it would
        // make this file test the sample count rather than the size gate.
        private const int Samples = 120;
        private const double StepSeconds = 15.0;

        /// <summary>The reported case: a channel at zero with microsecond-scale noise must stay silent.</summary>
        [Fact]
        public void NoiseAroundZero_IsNotATrend()
        {
            var rng = new Random(20260730);
            var values = new double[Samples];

            // Mostly exact zeros with the occasional 1e-5, drifting down — the shape that produced the
            // incident. Monotone enough to clear tau, far too small to matter.
            for (var i = 0; i < Samples; i++)
            {
                values[i] = rng.NextDouble() < 0.3 ? 1e-5 * (Samples - i) / Samples : 0.0;
            }

            var result = Detect(values, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
        }

        /// <summary>A window that never leaves zero has no scale under any rule and cannot be material.</summary>
        [Fact]
        public void AnIdenticallyZeroSeries_IsNotATrend()
        {
            var result = Detect(new double[Samples], TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
        }

        /// <summary>
        /// The case that must survive the fix: an error rate leaving zero. Its median is zero — the series is
        /// at zero for most of the window — so a rule that silenced zero-median series would miss it.
        /// </summary>
        [Fact]
        public void ASignalClimbingAwayFromZero_IsStillCaught()
        {
            var values = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                values[i] = i < Samples * 0.55 ? 0.0 : 0.05 * (i - (Samples * 0.55)) / (Samples * 0.45);
            }

            var result = Detect(values, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Rising, result.Direction);

            // The wording has to say which scale decided it — "of typical" would name a value the series
            // never held.
            Assert.Contains("peak", result.Reason, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>An ordinary non-zero series must be unaffected: the median still supplies the scale.</summary>
        [Fact]
        public void AnOrdinaryRisingSeries_StillReportsAgainstItsMedian()
        {
            var values = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                values[i] = 1000.0 + (i * 12.0);
            }

            var result = Detect(values, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Rising, result.Direction);
            Assert.Contains("typical", result.Reason, StringComparison.Ordinal);
        }

        /// <summary>
        /// The absolute gate is the caller's per-metric say in units the detector cannot know. Same series,
        /// same statistics — only the floor moves.
        /// </summary>
        [Fact]
        public void TheAbsoluteGate_SilencesAChangeTooSmallToActOn()
        {
            var values = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                values[i] = 1000.0 + (i * 12.0);   // ~700 units across the window
            }

            Assert.Equal(DetectionStatus.Anomalous, Detect(values, TrendOptions.Balanced).Status);

            var demanding = TrendOptions.Balanced with
            {
                MinAbsoluteChangeOverWindow = 5000.0
            };

            Assert.Equal(DetectionStatus.Healthy, Detect(values, demanding).Status);
        }

        /// <summary>Zero keeps meaning "gate disabled", so existing callers are unchanged by its addition.</summary>
        [Fact]
        public void TheAbsoluteGate_IsDisabledAtZero()
        {
            Assert.Equal(0.0, TrendOptions.Balanced.MinAbsoluteChangeOverWindow);
            Assert.True(TrendOptions.Balanced.IsValid);
            Assert.True((TrendOptions.Balanced with
            {
                MinAbsoluteChangeOverWindow = 0.25
            }).IsValid);
        }

        private static TrendResult Detect(double[] values, TrendOptions options)
        {
            var times = new double[values.Length];

            for (var i = 0; i < values.Length; i++)
            {
                times[i] = i * StepSeconds;
            }

            return new TrendDetector().Detect(values, times, options);
        }
    }
}
