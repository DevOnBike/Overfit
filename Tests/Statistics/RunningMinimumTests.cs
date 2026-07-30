// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Statistics
{
    /// <summary>
    /// Correctness for <see cref="RunningMinimum"/>, pinned against the naive definition it replaces.
    ///
    /// <para>The deque sweep is the kind of code that is right on the cases you thought of and wrong two
    /// samples later, so the load-bearing test here is not a hand-written expectation — it is a randomised
    /// comparison against the O(n·k) nested loop, which is obviously correct and far too slow to ship.</para>
    /// </summary>
    public sealed class RunningMinimumTests
    {
        [Fact]
        public void MatchesTheNaiveDefinition_AcrossLengthsAndLookbacks()
        {
            var rng = new Random(20260730);

            foreach (var length in new[] { 1, 2, 7, 64, 257 })
            {
                var source = new double[length];

                for (var i = 0; i < length; i++)
                {
                    source[i] = rng.NextDouble() * 1000.0;
                }

                foreach (var lookback in new[] { 1, 2, 3, 8, 33, 512 })
                {
                    var actual = new double[length];
                    var scratch = new int[RunningMinimum.RequiredScratchLength(length)];

                    RunningMinimum.Compute(source, lookback, actual, scratch);

                    for (var i = 0; i < length; i++)
                    {
                        Assert.Equal(NaiveMin(source, i, lookback), actual[i], 12);
                    }
                }
            }
        }

        [Fact]
        public void SkipsGaps_RatherThanPropagatingThem()
        {
            // A scrape gap is missing information about memory, not a claim that memory was unmeasurable.
            double[] source = [5.0, double.NaN, 3.0, double.NaN, double.NaN, 9.0];
            var actual = new double[source.Length];
            var scratch = new int[source.Length];

            RunningMinimum.Compute(source, 3, actual, scratch);

            Assert.Equal(5.0, actual[0]);
            Assert.Equal(5.0, actual[1]);
            Assert.Equal(3.0, actual[2]);
            Assert.Equal(3.0, actual[3]);
            Assert.Equal(3.0, actual[4]);

            // Position 5 looks back over 3, 4, 5 — only the 9 is finite there.
            Assert.Equal(9.0, actual[5]);
        }

        [Fact]
        public void ReportsNoEvidence_WhenAWindowHoldsNothingFinite()
        {
            double[] source = [1.0, double.NaN, double.NaN, double.NaN];
            var actual = new double[source.Length];
            var scratch = new int[source.Length];

            RunningMinimum.Compute(source, 2, actual, scratch);

            Assert.Equal(1.0, actual[1]);
            Assert.True(double.IsNaN(actual[2]));
            Assert.True(double.IsNaN(actual[3]));
        }

        /// <summary>
        /// The property the whole design rests on: over a full cycle the floor does not depend on where in the
        /// cycle you happen to be looking.
        /// </summary>
        [Fact]
        public void IsPhaseInvariant_OverAFullSawtoothCycle()
        {
            const int Period = 40;
            const int Samples = 400;

            var floors = new double[Samples];
            var scratch = new int[Samples];
            var sawtooth = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                sawtooth[i] = 1000.0 + (i % Period * 10.0);   // 1000 → 1390, reset, repeat
            }

            RunningMinimum.Compute(sawtooth, Period, floors, scratch);

            // Past the first full cycle every position reports the same floor, whatever the phase.
            for (var i = Period; i < Samples; i++)
            {
                Assert.Equal(1000.0, floors[i], 9);
            }

            // The instantaneous series, by contrast, spans the whole tooth — which is exactly the spread that
            // was making healthy replicas look like outliers.
            Assert.Equal(1390.0, sawtooth[Period - 1]);
            Assert.Equal(1000.0, sawtooth[Period]);
        }

        /// <summary>A leak lifts the floor; that is what makes the floor worth comparing at all.</summary>
        [Fact]
        public void TracksALeak_ThroughTheSawtooth()
        {
            const int Period = 40;
            const int Samples = 400;
            const double LeakPerSample = 2.0;

            var floors = new double[Samples];
            var scratch = new int[Samples];
            var leaking = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                leaking[i] = 1000.0 + (i * LeakPerSample) + (i % Period * 10.0);
            }

            RunningMinimum.Compute(leaking, Period, floors, scratch);

            Assert.True(floors[Samples - 1] > floors[Period],
                $"floor did not follow the leak: {floors[Period]} → {floors[Samples - 1]}");

            // The floor advances in steps, one per collection, so two positions a whole cycle apart differ by
            // exactly a cycle's worth of leak. Comparing arbitrary endpoints instead would compare where in the
            // step each happened to land, which is a property of the sampling, not of the leak.
            for (var i = Period; i + Period < Samples; i++)
            {
                Assert.Equal(Period * LeakPerSample, floors[i + Period] - floors[i], 6);
            }
        }

        [Fact]
        public void FloorWindow_MatchesTheEquivalentFullSweep()
        {
            const int Length = 300;
            const int Lookback = 57;
            const int WindowLength = 80;

            var rng = new Random(4242);
            var history = new double[Length];

            for (var i = 0; i < Length; i++)
            {
                history[i] = rng.NextDouble() * 500.0;
            }

            var reference = new double[Length];
            RunningMinimum.Compute(history, Lookback, reference, new int[Length]);

            for (var start = Lookback - 1; start + WindowLength <= Length; start += 37)
            {
                var window = new double[WindowLength];
                var scratch = new int[Length];

                Assert.True(RunningMinimum.TryFloorWindow(
                    history, start, WindowLength, Lookback, window, scratch));

                for (var i = 0; i < WindowLength; i++)
                {
                    Assert.Equal(reference[start + i], window[i], 12);
                }
            }
        }

        /// <summary>
        /// Refusing is the point: a floor taken over less than a full cycle carries the phase straight through
        /// and looks perfectly reasonable in a report.
        /// </summary>
        [Fact]
        public void FloorWindow_RefusesAWindowItCannotBackWithHistory()
        {
            var history = new double[100];
            var window = new double[20];
            var scratch = new int[100];

            Assert.False(RunningMinimum.TryFloorWindow(history, windowStart: 10, 20, lookback: 40, window, scratch));
            Assert.True(RunningMinimum.TryFloorWindow(history, windowStart: 39, 20, lookback: 40, window, scratch));

            // Past the end of the history is a refusal too, not an exception — a short read is a normal
            // condition when a pod is younger than the window.
            Assert.False(RunningMinimum.TryFloorWindow(history, windowStart: 90, 20, lookback: 40, window, scratch));
        }

        [Fact]
        public void RejectsUnusableArguments()
        {
            var source = new double[10];
            var scratch = new int[10];

            Assert.Throws<ArgumentOutOfRangeException>(
                () => RunningMinimum.Compute(source, 0, new double[10], scratch));

            Assert.Throws<ArgumentOutOfRangeException>(
                () => RunningMinimum.Compute(source, RunningMinimum.MaxLookbackSamples + 1, new double[10], scratch));

            Assert.Throws<ArgumentException>(
                () => RunningMinimum.Compute(source, 3, new double[9], scratch));

            Assert.Throws<ArgumentException>(
                () => RunningMinimum.Compute(source, 3, new double[10], new int[9]));
        }

        private static double NaiveMin(ReadOnlySpan<double> source, int index, int lookback)
        {
            var best = double.NaN;

            for (var i = Math.Max(0, index - lookback + 1); i <= index; i++)
            {
                if (double.IsFinite(source[i]) && (double.IsNaN(best) || source[i] < best))
                {
                    best = source[i];
                }
            }

            return best;
        }
    }
}
