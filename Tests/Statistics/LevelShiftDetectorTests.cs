// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// The step detector: what it must see, and — more importantly — what it must not.
    ///
    /// <para>It exists because a step is invisible to every other family, which was measured: on a window
    /// split evenly by a 2.5× step the trend detector recovered the slope fine and still returned
    /// <c>Healthy</c> at p = 0.0695, and a 10× step scored p = 0.0794 — <b>worse</b>, because Mann-Kendall's
    /// tau counts rank order and a step scores about 0.51 whatever its height.</para>
    /// </summary>
    public sealed class LevelShiftDetectorTests
    {
        private static readonly LevelShiftDetector Detector = new();

        [Fact]
        public void AStepInTheMiddleIsFound()
        {
            var result = Detector.Detect(Step(from: 100.0, to: 250.0, at: 40), LevelShiftOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Rising, result.Direction);
            Assert.InRange(result.RelativeChange, 1.4, 1.6);
            Assert.InRange(result.EffectSize, 0.99, 1.0);
        }

        /// <summary>
        /// A level that fell is as much a change as one that rose. A deployment that halved throughput is not
        /// healthy because the number went down.
        /// </summary>
        [Fact]
        public void AStepDownIsFoundToo()
        {
            var result = Detector.Detect(Step(from: 250.0, to: 100.0, at: 40), LevelShiftOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Falling, result.Direction);
            Assert.True(result.AbsoluteChange < 0.0);
        }

        [Fact]
        public void AFlatSeriesIsHealthy()
        {
            var result = Detector.Detect(Step(from: 100.0, to: 100.0, at: 40), LevelShiftOptions.Balanced);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Equal(0.0, result.Severity);
        }

        /// <summary>
        /// A steady climb is the trend family's job, and this one must not double-report it. The gate that
        /// separates them is the effect size: a ramp's halves overlap far more than a step's.
        /// </summary>
        [Fact]
        public void AGentleRampIsNotAStep()
        {
            var series = new double[80];

            for (var i = 0; i < series.Length; i++)
            {
                series[i] = 100.0 + (i * 0.15);
            }

            var result = Detector.Detect(series, LevelShiftOptions.Balanced);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
        }

        /// <summary>
        /// The size gates, which are the whole reason a rank test is not enough: a perfectly consistent 2%
        /// shift separates the halves at delta 1.00 and is nobody's problem.
        /// </summary>
        [Fact]
        public void AConsistentButTinyShiftIsNotWorthReporting()
        {
            var result = Detector.Detect(Step(from: 100.0, to: 102.0, at: 40), LevelShiftOptions.Balanced);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.InRange(result.EffectSize, 0.9, 1.0);
            Assert.Contains("below the 25%", result.Reason, StringComparison.Ordinal);
        }

        [Fact]
        public void TheAbsoluteFloorCanRefuseAProportionallyLargeStep()
        {
            // Two milliwatts of nothing doubling is still nothing, and only the caller knows that.
            var options = LevelShiftOptions.Balanced with { MinAbsoluteChange = 1000.0 };
            var result = Detector.Detect(Step(from: 100.0, to: 250.0, at: 40), options);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Contains("own units", result.Reason, StringComparison.Ordinal);
        }

        [Fact]
        public void TooFewSamplesIsNotHealth()
        {
            var result = Detector.Detect(new double[] { 1, 2, 3, 4 }, LevelShiftOptions.Balanced);

            Assert.Equal(DetectionStatus.InsufficientData, result.Status);
        }

        /// <summary>
        /// A series that starts at zero has no proportion to take. The relative gate cannot judge it, and the
        /// result must say so rather than silently reading "cannot measure" as "no change" — the mistake that
        /// cost a severity-0.59 incident over a few microseconds of GC on the trend side.
        /// </summary>
        [Fact]
        public void AShiftAwayFromZeroIsReportedAsImmeasurableInProportion()
        {
            var series = new double[80];

            for (var i = 40; i < series.Length; i++)
            {
                series[i] = 5.0;
            }

            var result = Detector.Detect(series, LevelShiftOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.True(double.IsPositiveInfinity(result.RelativeChange));
            Assert.Contains("started from zero", result.Reason, StringComparison.Ordinal);
        }

        [Fact]
        public void DefaultOptionsAreRejected()
        {
            Assert.Throws<ArgumentException>(() => Detector.Detect(new double[80], default));
        }

        /// <summary>Tight noise around each level, so the halves separate on the level and not on the noise.</summary>
        private static double[] Step(double from, double to, int at, int samples = 80)
        {
            var rng = new Random(20260801);
            var series = new double[samples];

            for (var i = 0; i < samples; i++)
            {
                var level = i < at ? from : to;

                series[i] = level * (0.99 + (0.02 * rng.NextDouble()));
            }

            return series;
        }
    }
}
