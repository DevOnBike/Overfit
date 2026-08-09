// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Why the learned seasonal history makes the guard noisier instead of quieter (<c>AN-F1</c>).
    ///
    /// <para><b>The measurement that opened this.</b> Replaying A1's own window cold reproduces its 11
    /// incidents exactly, so the replay is faithful. Feeding the deployed learned state into the same window
    /// takes it to <b>33 opened</b> and findings from 296 to 517. Stripping the payload one section at a
    /// time isolates it: calibration moves nothing, <b>the seasonal history is the whole lever</b>. The
    /// explanation on offer — that an hourly-anchored interpolation subtracted from a smooth signal injects
    /// apparent trend — was reasoning, not measurement.
    ///
    /// <para><b>Two mechanisms could produce it and they are distinguishable, which is what this exists to
    /// do.</b> Either the ANCHORS are noisy — each is the median across days of a single five-minute
    /// reading, so with two or three days it carries nearly the full per-sample error, and a line drawn
    /// between two wrong points has a slope of its own — or the SHAPE is wrong, because a straight chord
    /// subtracted from a curved signal leaves the curvature behind. The first gets better with more days;
    /// the second never does.</para>
    ///
    /// <para><b>Measured, and it is the second — which was not the leading hypothesis.</b> More days change
    /// nothing at all: 222 of 284 windows at two days, the same 222 at seven. What moves it is <b>signal
    /// noise</b>, and in the direction that first looks backwards — noise <i>hides</i> the effect. The
    /// reason is that the residual of a chord under a smooth curve is perfectly monotone inside a
    /// twenty-minute window, which is precisely the pattern a rank correlation is built to detect and be
    /// confident about; real scatter breaks the monotonicity that convinces it.</para>
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

        [LongFact("3s")]
        public void SubtractingAnHourlyAnchoredExpectationInjectsTrend()
        {
            var report = new StringBuilder();

            report.Append("A smooth diurnal signal, judged raw and judged after subtracting the seasonal\n");
            report.Append("expectation. Same windows, same detector, same options — the only difference is\n");
            report.Append("the subtraction.\n\n");
            // Run WHERE THE EFFECT LIVES. The first version swept history at noise 0.3, which suppresses
            // the effect entirely, so every row read zero and the table could not have distinguished
            // anything — a comparison in which no arm can score is not evidence about any arm.
            report.Append("Does MORE HISTORY help? (noise fixed at 0.01, where the effect is present)\n");
            report.Append($"   {"history",10}{"raw",10}{"adjusted",12}{"of windows",12}\n");

            for (var days = 2; days <= 7; days++)
            {
                var (raw, adjusted, windows) = Score(days, noise: 0.01);

                report.Append($"   {days + " days",10}{raw,10}{adjusted,12}{windows,12}\n");
            }

            report.Append("\nDoes SIGNAL NOISE? (history fixed at 3 days)\n");
            report.Append($"   {"noise",10}{"of amp",9}{"raw",10}{"adjusted",12}{"of windows",12}\n");

            foreach (var level in new[] { 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4 })
            {
                var (raw, adjusted, windows) = Score(3, level);

                report.Append($"   {level,10:F3}{level / Amplitude * 100.0,8:F1}%{raw,10}{adjusted,12}"
                              + $"{windows,12}\n");
            }

            report.Append("\nFalling with HISTORY would mean noisy anchors, and more days would fix it.\n");
            report.Append("Falling with SIGNAL NOISE means the interpolation SHAPE: a chord subtracted from a\n");
            report.Append("curve leaves a residual that is perfectly monotone inside a twenty-minute window —\n");
            report.Append("exactly what a rank test is built to find — and real scatter breaks the\n");
            report.Append("monotonicity that convinces it. More days will never help.\n");

            _output.WriteLine(report.ToString());

            Assert.True(WindowSamples > 1);
        }

        /// <summary>
        /// Builds <paramref name="days"/> days of history, then scores every window of the following day
        /// twice: on the raw series, and on the series minus the seasonal expectation.
        /// </summary>
        private static (int Raw, int Adjusted, int Windows) Score(int days, double noise)
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

            // Observed once per CADENCE, as the guard does — not once per scrape. That is what makes each
            // hourly anchor a single five-minute reading rather than an average of the hour.
            for (var i = 0; i < perDay * days; i += perCadence)
            {
                history.Observe(Workload, Signal, Start + (Scrape * i), series[i]);
            }

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

            // The day after the history, which is the situation the deployed guard is in.
            for (var offset = perDay * days; offset + WindowSamples < series.Length; offset += perCadence)
            {
                var windowStart = Start + (Scrape * offset);

                if (!history.TryExpectation(Workload, Signal, windowStart, Scrape, days, expectation))
                {
                    continue;
                }

                windows++;
                series.AsSpan(offset, WindowSamples).CopyTo(rawWindow);

                for (var i = 0; i < WindowSamples; i++)
                {
                    adjustedWindow[i] = rawWindow[i] - expectation[i];
                }

                raw += detector.Detect(rawWindow, times, TrendOptions.Balanced).Status
                    == DetectionStatus.Anomalous ? 1 : 0;

                adjusted += detector.Detect(adjustedWindow, times, TrendOptions.Balanced).Status
                    == DetectionStatus.Anomalous ? 1 : 0;
            }

            return (raw, adjusted, windows);
        }

        /// <summary>
        /// A smooth daily curve — the shape the seasonal baseline exists to cancel. Identical every day, so
        /// a perfect expectation would leave exactly zero and any finding is injected by the correction.
        /// </summary>
        private static double Value(int index, int perDay)
        {
            var phase = (index % perDay) / (double)perDay * 2.0 * Math.PI;

            return 10.0 + (Amplitude * Math.Sin(phase));
        }
    }
}
