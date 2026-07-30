// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    public sealed class TrendDetectorTests
    {
        private const double ScrapeSeconds = 30.0;

        private static readonly TrendDetector Detector = new();

        [Fact]
        public void AMemoryLeak_IsDetectedWithItsRateAndTimeToLimit()
        {
            // 20 MB every ten minutes on a 1 GB baseline, sampled for two hours — the case a
            // "memory > 80%" rule stays quiet about until it is far too late.
            const double bytesPerSecond = 20.0 * 1024 * 1024 / 600.0;
            var (values, times) = Series(240, seed: 1, start: 1024.0 * 1024 * 1024, slope: bytesPerSecond, noise: 0.01);

            var limit = 2.0 * 1024 * 1024 * 1024;
            var result = Detector.Detect(values, times, TrendOptions.Balanced, limit);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Rising, result.Direction);
            Assert.True(result.IsDecided);

            // Theil-Sen recovers the planted rate.
            Assert.Equal(bytesPerSecond, result.SlopePerSecond, bytesPerSecond * 0.1);

            // Projected from the END of the window, not its start: the two hours observed have already eaten
            // ~250 MB of the 1 GB headroom, leaving ~823 MB at 20 MB per 10 minutes — about 6.5 hours.
            Assert.NotNull(result.TimeToLimit);
            Assert.InRange(result.TimeToLimit!.Value.TotalHours, 6.0, 7.0);
            Assert.Contains("limit is reached in", result.Reason);
        }

        [Fact]
        public void AFlatNoisySeries_IsHealthy()
        {
            var (values, times) = Series(240, seed: 2, start: 400.0, slope: 0.0, noise: 0.05);

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Equal(TrendDirection.None, result.Direction);
            Assert.Null(result.TimeToLimit);
        }

        [Fact]
        public void ADecline_IsReportedRatherThanIgnored()
        {
            // A collapsing cache hit ratio or request rate is a symptom too.
            var (values, times) = Series(200, seed: 3, start: 1000.0, slope: -0.5, noise: 0.01);

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Falling, result.Direction);
            Assert.True(result.KendallTau < -0.5, $"tau = {result.KendallTau}");
            Assert.True(result.SlopePerSecond < 0.0);
        }

        [Fact]
        public void SpikesDoNotSteerTheSlope()
        {
            // Theil-Sen's whole reason for being: a handful of scrape artefacts must not become the trend.
            var (values, times) = Series(200, seed: 4, start: 500.0, slope: 0.0, noise: 0.02);
            values[40] = 50_000.0;
            values[41] = 60_000.0;
            values[150] = 90_000.0;

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.True(Math.Abs(result.SlopePerSecond) < 0.05, $"slope = {result.SlopePerSecond}");
        }

        [Fact]
        public void ASmoothRandomWalk_IsNotReportedAsATrend()
        {
            // The failure that makes naive Mann-Kendall unusable on monitoring data. A random walk has no
            // trend, but consecutive samples are almost identical, so an uncorrected test sees overwhelming
            // "evidence" of monotone movement. The autocorrelation correction is what stops this.
            var rng = new Random(20260724);
            var values = new double[300];
            var times = new double[300];
            var level = 400.0;

            for (var i = 0; i < values.Length; i++)
            {
                level += (rng.NextDouble() - 0.5) * 4.0;
                values[i] = level;
                times[i] = i * ScrapeSeconds;
            }

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.True(result.Autocorrelation > 0.5,
                $"fixture is not autocorrelated enough to prove anything: rho = {result.Autocorrelation}");
            Assert.Equal(DetectionStatus.Healthy, result.Status);
        }

        [Fact]
        public void AConsistentButTinyClimb_IsNotWorthWaking()
        {
            // Perfectly monotone — tau is essentially 1.0 — but it moves 0.5% across the window. Significance
            // alone would fire; the relative-change gate is what stops it.
            var (values, times) = Series(200, seed: 5, start: 1000.0, slope: 0.0008, noise: 0.0);

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.True(result.KendallTau > 0.95, $"tau = {result.KendallTau}");
            Assert.Equal(DetectionStatus.Healthy, result.Status);
            Assert.Contains("too small", result.Reason);
        }

        [Fact]
        public void StrictProfile_DemandsABiggerMoveThanBalanced()
        {
            // 12% across the window: past Balanced's 10%, short of Strict's 20%.
            var (values, times) = Series(200, seed: 6, start: 1000.0, slope: 0.02, noise: 0.005);

            var balanced = Detector.Detect(values, times, TrendOptions.Balanced);
            var strict = Detector.Detect(values, times, TrendOptions.Strict);

            Assert.Equal(DetectionStatus.Anomalous, balanced.Status);
            Assert.Equal(DetectionStatus.Healthy, strict.Status);
        }

        [Fact]
        public void TooShortAWindow_IsWarmingUp_NotInsufficientData()
        {
            var (values, times) = Series(10, seed: 7, start: 400.0, slope: 1.0, noise: 0.01);

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.WarmingUp, result.Status);
            Assert.False(result.IsHealthy);
            Assert.False(result.IsDecided);
            Assert.Equal(10, result.SampleCount);
            Assert.Contains("30 are required", result.Reason);
        }

        [Fact]
        public void NothingUsable_IsInsufficientData()
        {
            var values = new double[50];
            var times = new double[50];
            Array.Fill(values, double.NaN);
            for (var i = 0; i < times.Length; i++)
            {
                times[i] = i * ScrapeSeconds;
            }

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.InsufficientData, result.Status);
            Assert.Equal(0, result.SampleCount);

            var empty = Detector.Detect([], [], TrendOptions.Balanced);
            Assert.Equal(DetectionStatus.InsufficientData, empty.Status);
        }

        [Fact]
        public void RepeatedAndOutOfOrderTimestamps_AreDropped()
        {
            var (values, times) = Series(120, seed: 8, start: 400.0, slope: 0.05, noise: 0.01);
            times[30] = times[29];      // duplicated scrape
            times[60] = times[10];      // clock went backwards

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(118, result.SampleCount);
            Assert.Equal(DetectionStatus.Anomalous, result.Status);
        }

        [Fact]
        public void NoLimitSupplied_MeansNoProjection()
        {
            var (values, times) = Series(200, seed: 9, start: 400.0, slope: 0.05, noise: 0.01);

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Null(result.TimeToLimit);
        }

        [Fact]
        public void MovingAwayFromTheLimit_ProjectsNothing()
        {
            var (values, times) = Series(200, seed: 10, start: 800.0, slope: -0.05, noise: 0.01);

            var result = Detector.Detect(values, times, TrendOptions.Balanced, limit: 2000.0);

            Assert.Equal(TrendDirection.Falling, result.Direction);
            Assert.Null(result.TimeToLimit);
        }

        [Fact]
        public void AZeroSeriesWithARealClimb_IsScaledAgainstItsPeakInsteadOfDividingByNothing()
        {
            // Error counts sitting at zero and then walking upward: the median is 0, so "10% of typical" is
            // meaningless. The detector must still report the climb, and name the scale it used instead.
            //
            // This test previously asserted that the size gate was SKIPPED when the median was zero. That was
            // the behaviour, and it was wrong in the other direction: skipping the gate meant "the size
            // cannot be judged" counted as "the size is large", which on the cluster lab raised an incident
            // over GcPauseRatio noise of about 10^-5. The gate now falls back to the largest magnitude the
            // window reached, which keeps this case — a real climb away from zero — and drops that one.
            var values = new double[120];
            var times = new double[120];
            for (var i = 0; i < values.Length; i++)
            {
                values[i] = i < 60 ? 0.0 : (i - 60) * 0.25;
                times[i] = i * ScrapeSeconds;
            }

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.Equal(TrendDirection.Rising, result.Direction);
            Assert.Contains("the window's peak", result.Reason, StringComparison.Ordinal);
            Assert.Contains("no typical", result.Reason, StringComparison.Ordinal);
        }

        [Fact]
        public void LongWindowsAreThinnedRatherThanRefused()
        {
            var (values, times) = Series(5000, seed: 11, start: 1000.0, slope: 0.01, noise: 0.01);

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(DetectionStatus.Anomalous, result.Status);
            Assert.True(result.SampleCount <= 600, $"expected thinning, got {result.SampleCount} samples");
            Assert.Equal(0.01, result.SlopePerSecond, 0.002);
        }

        [Fact]
        public void MismatchedOrInvalidInput_Throws()
        {
            Assert.Throws<ArgumentException>(() =>
                Detector.Detect(new double[5], new double[4], TrendOptions.Balanced));

            Assert.Throws<ArgumentException>(() =>
                Detector.Detect(new double[50], new double[50], default));

            Assert.False(default(TrendOptions).IsValid);
        }

        [Fact]
        public void ProjectedChangeOver_UsesTheFittedRate()
        {
            var (values, times) = Series(200, seed: 12, start: 400.0, slope: 0.5, noise: 0.005);

            var result = Detector.Detect(values, times, TrendOptions.Balanced);

            Assert.Equal(result.SlopePerSecond * 3600.0, result.ProjectedChangeOver(3600.0), 9);
            Assert.Equal(0.5 * 3600.0, result.ProjectedChangeOver(3600.0), 100.0);
        }

        /// <summary>
        /// A series with a planted slope and multiplicative noise, at a fixed scrape interval. Noise is
        /// deterministic per seed so effect sizes are reproducible.
        /// </summary>
        private static (double[] Values, double[] Times) Series(
            int count,
            int seed,
            double start,
            double slope,
            double noise)
        {
            var rng = new Random(seed);
            var values = new double[count];
            var times = new double[count];

            for (var i = 0; i < count; i++)
            {
                var t = i * ScrapeSeconds;
                times[i] = t;
                values[i] = (start + (slope * t)) * (1.0 + ((rng.NextDouble() - 0.5) * 2.0 * noise));
            }

            return (values, times);
        }
    }
}
