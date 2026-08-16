// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// What the seasonal correction does to a signal that has nothing wrong with it — and what the naive
    /// version of it would have done.
    ///
    /// <para><b>Why this exists.</b> Replaying A1's window cold reproduces its 11 incidents exactly, so the
    /// replay is faithful. Feeding the deployed learned state into the same window takes it to <b>33</b>, and
    /// stripping the payload section by section isolates the seasonal history as the whole lever
    /// (<c>AN-F1</c>). The standing explanation was that an hourly-anchored interpolation subtracted from a
    /// smooth signal injects apparent trend. <b>Measured here, against the guard's actual arithmetic, it does
    /// not</b> — 0 findings in every cell, at every history depth from two to seven days and every noise
    /// level from a perfectly smooth curve to 10% of amplitude. That whole family of explanations is out.</para>
    ///
    /// <para><b>The load-bearing line is <c>+ level</c> in <see cref="Incidents.AnomalyGuard"/>'s
    /// <c>Adjust</c>, and this measures how load-bearing.</b> Subtracting the expectation alone leaves a
    /// residual centred on zero: the scale the relative gates are percentages of collapses by <b>65x at the
    /// lab's own scatter</b> and 786x on a quiet signal, and on a smooth curve the leftover curvature is
    /// perfectly monotone inside a twenty-minute window — which is exactly the pattern a rank test is built
    /// to be most confident about, giving 222 findings in 284 windows where the raw series gives none.
    /// Adding the median back removes both, completely.</para>
    ///
    /// <para><b>Written after getting it wrong.</b> The first version of this diagnostic omitted the
    /// add-back, reproduced those two effects, and they were recorded as the diagnosis of <c>AN-F1</c> before
    /// anyone checked what <c>Adjust</c> actually computes. Keeping both arms here is the correction: the
    /// naive column is what a reviewer would predict, the guard column is what the guard does, and the gap
    /// between them is a design decision that was previously defended only by a comment.</para>
    /// </summary>
    public sealed class SeasonalExpectationNoiseDiagnostics
    {
        private const string Workload = "lab/lab-workload";
        private const MetricIndex Signal = MetricIndex.RequestsPerSecond;

        /// <summary>
        /// The SCRAPE interval, which is what a window is made of. Distinct from the cadence, and the first
        /// version of this diagnostic conflated the two: it built four-sample windows at the cadence and
        /// every arm scored zero, because four points cannot carry a significant rank correlation. The guard
        /// judges eighty.
        /// </summary>
        private static readonly TimeSpan Scrape = TimeSpan.FromSeconds(15);

        /// <summary>How often a window is evaluated, and how often history is observed.</summary>
        private static readonly TimeSpan Cadence = TimeSpan.FromMinutes(5);

        /// <summary>Twenty minutes of 15-second scrapes — the guard's default evaluation window.</summary>
        private const int WindowSamples = 80;

        /// <summary>Amplitude of the synthetic daily curve, so noise can be quoted as a share of it.</summary>
        private const double Amplitude = 4.0;

        private static readonly DateTimeOffset Start = new(2026, 8, 1, 0, 0, 0, TimeSpan.Zero);

        private readonly ITestOutputHelper _output;

        public SeasonalExpectationNoiseDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact("4s")]
        public void TheSeasonalCorrectionInjectsNothingAndTheMedianAddBackIsWhy()
        {
            var report = new StringBuilder();

            report.Append("A smooth diurnal signal with nothing wrong. Raw, then corrected two ways:\n");
            report.Append("  guard  = series - expectation + median(expectation)   <- AnomalyGuard.Adjust\n");
            report.Append("  naive  = series - expectation                         <- the obvious version\n\n");

            report.Append("Across SIGNAL NOISE, 3 days of history:\n");
            report.Append($"   {"noise",9}{"of amp",8}{"raw",8}{"guard",8}{"naive",8}{"windows",10}\n");

            foreach (var level in new[] { 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4 })
            {
                var guard = Score(3, level, addBack: true);
                var naive = Score(3, level, addBack: false);

                report.Append($"   {level,9:F3}{level / Amplitude * 100.0,7:F1}%{guard.Raw,8}"
                              + $"{guard.Adjusted,8}{naive.Adjusted,8}{guard.Windows,10}\n");
            }

            report.Append("\nAcross HISTORY DEPTH, noise 0.01:\n");
            report.Append($"   {"history",9}{"raw",8}{"guard",8}{"naive",8}{"windows",10}\n");

            for (var days = 2; days <= 7; days++)
            {
                var guard = Score(days, 0.01, addBack: true);
                var naive = Score(days, 0.01, addBack: false);

                report.Append($"   {days + "d",9}{guard.Raw,8}{guard.Adjusted,8}{naive.Adjusted,8}"
                              + $"{guard.Windows,10}\n");
            }

            report.Append("\nThe scale every relative gate is a percentage of:\n");
            report.Append($"   {"noise",9}{"of amp",8}{"raw",12}{"guard",12}{"naive",12}{"collapse",11}\n");

            foreach (var level in new[] { 0.01, 0.1, 0.3, 0.5 })
            {
                var (raw, guard, naive) = Scale(3, level);
                var collapse = naive > 0.0 ? raw / naive : double.NaN;

                report.Append($"   {level,9:F3}{level / Amplitude * 100.0,7:F1}%{raw,12:F3}{guard,12:F3}"
                              + $"{naive,12:F3}{collapse,10:F1}x\n");
            }

            report.Append("\nThe guard column is the product. Zero findings everywhere, scale preserved —\n");
            report.Append("so the correction does NOT explain AN-F1's 11 -> 33, and the family of\n");
            report.Append("explanations built on it is eliminated.\n");
            report.Append("The naive column is what one line of arithmetic prevents.\n");

            _output.WriteLine(report.ToString());

            // The claim in the doc comment, pinned: the add-back is load-bearing, not decoration.
            var withAddBack = Score(3, 0.0, addBack: true);
            var without = Score(3, 0.0, addBack: false);

            Assert.Equal(0, withAddBack.Adjusted);
            Assert.True(without.Adjusted > 100,
                $"the naive form should be badly wrong on a smooth signal; got {without.Adjusted}");
        }

        /// <summary>
        /// Builds <paramref name="days"/> days of history, then scores every window of the following day
        /// raw and corrected.
        /// </summary>
        private static (int Raw, int Adjusted, int Windows) Score(int days, double noise, bool addBack)
        {
            var (series, history, perDay, perCadence) = Build(days, noise);
            var detector = new TrendDetector();
            var times = new double[WindowSamples];

            for (var i = 0; i < WindowSamples; i++)
            {
                times[i] = i * Scrape.TotalSeconds;
            }

            var rawWindow = new double[WindowSamples];
            var expectation = new double[WindowSamples];
            var adjustedWindow = new double[WindowSamples];

            var raw = 0;
            var adjusted = 0;
            var windows = 0;

            for (var offset = perDay * days; offset + WindowSamples < series.Length; offset += perCadence)
            {
                if (!history.TryExpectation(
                        Workload, Signal, Start + (Scrape * offset), Scrape, days, expectation))
                {
                    continue;
                }

                windows++;
                series.AsSpan(offset, WindowSamples).CopyTo(rawWindow);

                var level = addBack ? Median(expectation) : 0.0;

                for (var i = 0; i < WindowSamples; i++)
                {
                    adjustedWindow[i] = rawWindow[i] - expectation[i] + level;
                }

                raw += detector.Detect(rawWindow, times, TrendOptions.Balanced).Status
                    == DetectionStatus.Anomalous ? 1 : 0;

                adjusted += detector.Detect(adjustedWindow, times, TrendOptions.Balanced).Status
                    == DetectionStatus.Anomalous ? 1 : 0;
            }

            return (raw, adjusted, windows);
        }

        /// <summary>Typical magnitude per window — raw, as the guard corrects it, and naively.</summary>
        private static (double Raw, double Guard, double Naive) Scale(int days, double noise)
        {
            var (series, history, perDay, perCadence) = Build(days, noise);
            var expectation = new double[WindowSamples];
            var raws = new List<double>();
            var guards = new List<double>();
            var naives = new List<double>();

            for (var offset = perDay * days; offset + WindowSamples < series.Length; offset += perCadence)
            {
                if (!history.TryExpectation(
                        Workload, Signal, Start + (Scrape * offset), Scrape, days, expectation))
                {
                    continue;
                }

                var window = series.AsSpan(offset, WindowSamples);
                var level = Median(expectation);
                var raw = new double[WindowSamples];
                var guard = new double[WindowSamples];
                var naive = new double[WindowSamples];

                for (var i = 0; i < WindowSamples; i++)
                {
                    raw[i] = Math.Abs(window[i]);
                    guard[i] = Math.Abs(window[i] - expectation[i] + level);
                    naive[i] = Math.Abs(window[i] - expectation[i]);
                }

                Array.Sort(raw);
                Array.Sort(guard);
                Array.Sort(naive);
                raws.Add(raw[WindowSamples / 2]);
                guards.Add(guard[WindowSamples / 2]);
                naives.Add(naive[WindowSamples / 2]);
            }

            raws.Sort();
            guards.Sort();
            naives.Sort();

            return raws.Count == 0
                ? (double.NaN, double.NaN, double.NaN)
                : (raws[raws.Count / 2], guards[guards.Count / 2], naives[naives.Count / 2]);
        }

        private static (double[] Series, MetricHistory History, int PerDay, int PerCadence) Build(
            int days, double noise)
        {
            var history = new MetricHistory();
            var rng = new Random(20260809);
            var perDay = (int)(TimeSpan.FromDays(1) / Scrape);
            var perCadence = (int)(Cadence / Scrape);
            var series = new double[perDay * (days + 1)];

            for (var i = 0; i < series.Length; i++)
            {
                series[i] = Value(i, perDay) + ((rng.NextDouble() - 0.5) * 2.0 * noise);
            }

            // Once per CADENCE, as the guard does — which is what makes each hourly anchor a single
            // five-minute reading rather than an average of the hour.
            for (var i = 0; i < perDay * days; i += perCadence)
            {
                history.Observe(Workload, Signal, Start + (Scrape * i), series[i]);
            }

            return (series, history, perDay, perCadence);
        }

        /// <summary>The median, as <c>AnomalyGuard.Adjust</c> takes it.</summary>
        private static double Median(double[] values)
        {
            var copy = (double[])values.Clone();

            Array.Sort(copy);

            return copy[copy.Length / 2];
        }

        /// <summary>
        /// A smooth daily curve — the shape the seasonal baseline exists to cancel. Identical every day, so
        /// a perfect correction leaves exactly zero and any finding is manufactured by the correction.
        /// </summary>
        private static double Value(int index, int perDay)
        {
            var phase = (index % perDay) / (double)perDay * 2.0 * Math.PI;

            return 10.0 + (Amplitude * Math.Sin(phase));
        }
    }
}
