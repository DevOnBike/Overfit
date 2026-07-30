// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// The seasonal expectation and the residual test built on it.
    ///
    /// <para>The measurement that motivated this: on a healthy synthetic population with a four-hour window the
    /// raw trend test produced <b>2551 false incidents a day</b>, almost all of them the daily traffic curve.
    /// With the residual, 376 — a 6.8x reduction, and <c>RequestsPerSecond</c> leaves the top signals entirely.
    /// These tests pin the mechanism behind that.</para>
    /// </summary>
    public sealed class SeasonalBaselineTests
    {
        private const int PerPeriod = 96;   // a "day" of 96 samples keeps the fixtures readable

        private static readonly TrendDetector Detector = new();

        [Fact]
        public void WithoutEnoughHistory_ItSaysSoRatherThanGuessing()
        {
            // One period of history is not a season, and treating it as one would make the first day of
            // operation the definition of normal.
            // Three periods, so a window can still sit after the second one — with two the second assertion
            // would run off the end of the history and throw for a reason that has nothing to do with the point.
            var history = Cycle(periods: 3, amplitude: 100.0, drift: 0.0);
            var expectation = new double[24];

            // One whole period precedes the window, so three cannot be satisfied.
            Assert.False(SeasonalBaseline.TryBuild(
                history, windowStart: PerPeriod, windowLength: 24, PerPeriod, expectation, minimumPeriods: 3));

            // Two do fit once the window sits after the second period.
            Assert.True(SeasonalBaseline.TryBuild(
                history, windowStart: 2 * PerPeriod, windowLength: 24, PerPeriod, expectation, minimumPeriods: 2));

            // Below the supported floor is a contract violation, not a quiet "no".
            Assert.Throws<ArgumentOutOfRangeException>(() => SeasonalBaseline.TryBuild(
                history, windowStart: 2 * PerPeriod, windowLength: 24, PerPeriod, expectation, minimumPeriods: 1));
        }

        [Fact]
        public void TheExpectationReproducesTheCycle()
        {
            var history = Cycle(periods: 4, amplitude: 100.0, drift: 0.0);
            var start = 3 * PerPeriod;
            var expectation = new double[PerPeriod];

            Assert.True(SeasonalBaseline.TryBuild(history, start, PerPeriod, PerPeriod, expectation, minimumPeriods: 3));

            for (var i = 0; i < PerPeriod; i++)
            {
                Assert.Equal(history[start + i], expectation[i], 9);
            }
        }

        [Fact]
        public void OneBadDayDoesNotBecomeTheExpectation()
        {
            // A deploy, a load test, an incident: a mean would carry it forward and suppress detection for a
            // week. The median over three periods absorbs it.
            var history = Cycle(periods: 4, amplitude: 100.0, drift: 0.0);

            for (var i = 0; i < PerPeriod; i++)
            {
                history[PerPeriod + i] *= 8.0;
            }

            var start = 3 * PerPeriod;
            var expectation = new double[PerPeriod];

            Assert.True(SeasonalBaseline.TryBuild(history, start, PerPeriod, PerPeriod, expectation, minimumPeriods: 3));

            for (var i = 0; i < PerPeriod; i++)
            {
                Assert.Equal(history[start + i], expectation[i], 9);
            }
        }

        [Fact]
        public void APhaseWithNoHistory_IsUnknownNotZero()
        {
            var history = Cycle(periods: 4, amplitude: 100.0, drift: 0.0);
            var start = 3 * PerPeriod;

            // Wipe one phase across every previous period.
            for (var period = 0; period < 3; period++)
            {
                history[(period * PerPeriod) + 10] = double.NaN;
            }

            var expectation = new double[PerPeriod];
            Assert.True(SeasonalBaseline.TryBuild(history, start, PerPeriod, PerPeriod, expectation, minimumPeriods: 3));

            Assert.True(double.IsNaN(expectation[10]), "a phase with no usable history must be unknown, not zero");
            Assert.False(double.IsNaN(expectation[11]));
        }

        [Fact]
        public void TheRisingLimbOfACycle_IsATrendRawAndNothingSeasonally()
        {
            // The false positive the whole feature exists to remove: a window sitting on the rising limb is a
            // huge, perfectly monotone, entirely normal movement.
            var history = Cycle(periods: 4, amplitude: 400.0, drift: 0.0);
            var start = 3 * PerPeriod;                 // fourth period
            var window = PerPeriod / 4;                // a quarter of the cycle

            // Phase 0 to 0.25: sin rises monotonically from 0 to 1. Starting an eighth of a period later would
            // straddle the peak, where the series rises and comes back and the net slope is nothing — which is
            // how the first version of this fixture managed to prove the opposite of its own point.
            var offset = start;

            var values = new double[window];
            Array.Copy(history, offset, values, 0, window);

            var times = new double[window];
            for (var i = 0; i < window; i++)
            {
                times[i] = i * 60.0;
            }

            var expectation = new double[window];
            Assert.True(SeasonalBaseline.TryBuild(history, offset, window, PerPeriod, expectation, minimumPeriods: 3));

            var raw = Detector.Detect(values, times, TrendOptions.Balanced with { MinimumSamples = 12 });
            var seasonal = Detector.Detect(
                values, times, TrendOptions.Balanced with { MinimumSamples = 12 }, double.NaN, expectation);

            Assert.Equal(DetectionStatus.Anomalous, raw.Status);
            Assert.Equal(TrendDirection.Rising, raw.Direction);

            Assert.Equal(DetectionStatus.Healthy, seasonal.Status);
            Assert.Equal(TrendDirection.None, seasonal.Direction);
        }

        [Fact]
        public void ARealDriftOnTopOfTheCycle_IsStillFound()
        {
            // The other half of the contract: removing the season must not remove the leak. Same cycle, plus a
            // genuine upward drift that was not there on previous days.
            var history = Cycle(periods: 4, amplitude: 400.0, drift: 0.0);
            var start = 3 * PerPeriod;
            var window = PerPeriod / 2;

            for (var i = 0; i < window; i++)
            {
                history[start + i] += 12.0 * i;
            }

            var values = new double[window];
            Array.Copy(history, start, values, 0, window);

            var times = new double[window];
            for (var i = 0; i < window; i++)
            {
                times[i] = i * 60.0;
            }

            var expectation = new double[window];
            Assert.True(SeasonalBaseline.TryBuild(history, start, window, PerPeriod, expectation, minimumPeriods: 3));

            var seasonal = Detector.Detect(
                values, times, TrendOptions.Balanced with { MinimumSamples = 12 }, double.NaN, expectation);

            Assert.Equal(DetectionStatus.Anomalous, seasonal.Status);
            Assert.Equal(TrendDirection.Rising, seasonal.Direction);
        }

        [Fact]
        public void MisalignedExpectation_Throws()
        {
            Assert.Throws<ArgumentException>(() => Detector.Detect(
                new double[40], new double[40], TrendOptions.Balanced, double.NaN, new double[39]));
        }

        [Fact]
        public void SamplesPerPeriod_ConvertsWallClockToSamples()
        {
            Assert.Equal(5760, SeasonalBaseline.SamplesPerPeriod(TimeSpan.FromHours(24), TimeSpan.FromSeconds(15)));
            Assert.Equal(2880, SeasonalBaseline.SamplesPerPeriod(TimeSpan.FromHours(24), TimeSpan.FromSeconds(30)));
        }

        /// <summary>A clean sinusoidal cycle, optionally with a linear drift laid over the whole history.</summary>
        private static double[] Cycle(int periods, double amplitude, double drift)
        {
            var values = new double[periods * PerPeriod];

            for (var t = 0; t < values.Length; t++)
            {
                var phase = (t % PerPeriod) / (double)PerPeriod;
                values[t] = 1000.0 + (amplitude * Math.Sin(2.0 * Math.PI * phase)) + (drift * t);
            }

            return values;
        }
    }
}
