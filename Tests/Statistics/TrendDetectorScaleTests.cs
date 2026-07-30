// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// Which series the materiality gate measures against when an expectation is supplied.
    ///
    /// <para><b>The regression these pin shipped for part of a day.</b> With an expectation, the tested
    /// series is <c>observed − expected</c>, whose median sits near zero by construction. The gate was
    /// dividing by <i>that</i> median instead of the original series', so any change became an enormous
    /// percentage and the threshold passed everything. It showed up on the cluster lab as "rose by 2429% of
    /// typical" — a number nobody could read, produced by a gate that had stopped gating.</para>
    /// </summary>
    public sealed class TrendDetectorScaleTests
    {
        private const int Samples = 120;
        private const double StepSeconds = 15.0;

        /// <summary>
        /// A pod drifting a little against a steady group. Against the original series that is a few percent
        /// and below the gate; against the residual it would look enormous.
        /// </summary>
        [Fact]
        public void AResidualIsMeasuredAgainstTheOriginalSeriesScale()
        {
            var values = new double[Samples];
            var expectation = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                // Level 1000, drifting up by 0.02 per sample — 2.4 across the window, 0.24% of typical.
                expectation[i] = 1000.0;
                values[i] = 1000.0 + (i * 0.02);
            }

            var result = Detect(values, expectation);

            // Consistent and significant, but far too small to matter against a level of 1000.
            Assert.Equal(DetectionStatus.Healthy, result.Status);
        }

        /// <summary>The same drift without an expectation must reach the same verdict.</summary>
        [Fact]
        public void TheVerdictDoesNotDependOnWhetherAnExpectationWasSupplied()
        {
            var values = new double[Samples];
            var expectation = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                expectation[i] = 1000.0;
                values[i] = 1000.0 + (i * 0.02);
            }

            Assert.Equal(Detect(values).Status, Detect(values, expectation).Status);
        }

        /// <summary>
        /// And a drift that <i>is</i> material must still be caught through the residual — the fix must not
        /// have been a blanket silencing.
        /// </summary>
        [Fact]
        public void AMaterialDriftIsStillCaughtThroughTheResidual()
        {
            var values = new double[Samples];
            var expectation = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                expectation[i] = 1000.0;
                values[i] = 1000.0 + (i * 5.0);   // 595 across the window, ~50% of typical
            }

            var result = Detect(values, expectation);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Rising, result.Direction);

            // The reported size is against the original level, so it stays in a range a human can read.
            Assert.Contains("%", result.Reason, StringComparison.Ordinal);
            Assert.DoesNotContain("00,0%", result.Reason, StringComparison.Ordinal);
        }

        private static TrendResult Detect(double[] values, double[]? expectation = null)
        {
            var times = new double[values.Length];

            for (var i = 0; i < values.Length; i++)
            {
                times[i] = i * StepSeconds;
            }

            return new TrendDetector().Detect(
                values, times, TrendOptions.Balanced, double.NaN, expectation ?? []);
        }
    }
}
