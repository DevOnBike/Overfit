// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Why a step change shared by every replica is invisible, and whether splitting the window sees it.
    ///
    /// <para><b>The detection matrix left one row undetected by every family: CPU rising 2.5× on all twelve
    /// replicas at once.</b> Two thirds of that are structural and expected — peer comparison has no outlier
    /// when everybody moves together, and no absolute rule is configured for CPU. The third is the one worth
    /// establishing rather than assuming: the trend family sees the step in the windows that straddle it, and
    /// reports nothing.</para>
    ///
    /// <para><b>The hypothesis under test is that this is not a threshold problem but the estimator working as
    /// designed.</b> Theil-Sen takes the median of all pairwise slopes precisely so that a minority of
    /// discordant points cannot drag the fit. In a window split evenly by a step, most pairs sit <i>within</i>
    /// one level and carry a slope of about zero, so the median slope stays near zero no matter how large the
    /// step is. Robustness to a level shift and blindness to a level shift are the same property.</para>
    ///
    /// <para>If that holds, no tuning fixes it and the remedy has to be a different question: <b>split the
    /// window and rank-test the halves.</b> That is measured here alongside, on the same data, before anything
    /// is built.</para>
    /// </summary>
    public sealed class LevelShiftDiagnostics
    {
        private const double ScrapeSeconds = 15.0;
        private const int Samples = 80;

        private readonly ITestOutputHelper _output;

        public LevelShiftDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact("10ms")]
        public void ShowsThatAStepIsNotATrend()
        {
            var report = new StringBuilder();

            report.Append($"one {Samples}-sample window (20 min at {ScrapeSeconds:F0} s), 12 replicas\n");
            report.Append("the step lands where the column says; 'ramp' climbs across the whole window\n\n");

            report.Append($"   {"shape",-26}{"trend",-16}{"tau",8}{"slope*win",12}{"p",11}"
                          + $"   {"split-half",-12}{"delta",8}{"p",11}\n");

            Row(report, "step 2.5x at the middle", Step(2.5, at: Samples / 2));
            Row(report, "step 2.5x at 1/4", Step(2.5, at: Samples / 4));
            Row(report, "step 2.5x at 3/4", Step(2.5, at: 3 * Samples / 4));
            Row(report, "step 10x at the middle", Step(10.0, at: Samples / 2));
            Row(report, "ramp to 2.5x", Ramp(2.5));
            Row(report, "flat (control)", Step(1.0, at: Samples / 2));

            report.Append("\ntrend       what TrendDetector says about the cross-peer common component\n");
            report.Append("split-half  a Mann-Whitney test of the window's first half against its second\n");

            _output.WriteLine(report.ToString());

            Assert.True(true);
        }

        private static void Row(StringBuilder report, string title, double[] common)
        {
            var times = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                times[i] = i * ScrapeSeconds;
            }

            var verdict = new TrendDetector().Detect(common, times, TrendOptions.Balanced);
            var windowSeconds = times[times.Length - 1] - times[0];

            var half = Samples / 2;
            var comparison = MannWhitneyComparer.Instance.Compare(
                common.AsSpan(0, half), common.AsSpan(half, Samples - half));

            var sees = comparison.PValueCandidateWorse < 0.05 && Math.Abs(comparison.EffectSize) >= 0.33;

            report.Append($"   {title,-26}{verdict.Status,-16}{verdict.KendallTau,8:F2}"
                          + $"{verdict.SlopePerSecond * windowSeconds,12:G4}{verdict.PValue,11:G3}"
                          + $"   {(sees ? "SEES IT" : "-"),-12}"
                          + $"{Math.Abs(comparison.EffectSize),8:F2}{comparison.PValueCandidateWorse,11:G3}\n");
        }

        /// <summary>
        /// The cross-peer common component of twelve replicas that all step together — which is exactly what
        /// <c>CrossPeerBaseline</c> hands the trend detector, so this is the series the guard actually judges.
        /// </summary>
        private static double[] Step(double factor, int at)
        {
            var series = Base();

            for (var i = at; i < Samples; i++)
            {
                series[i] *= factor;
            }

            return series;
        }

        private static double[] Ramp(double factor)
        {
            var series = Base();

            for (var i = 0; i < Samples; i++)
            {
                series[i] *= 1.0 + ((factor - 1.0) * i / (Samples - 1.0));
            }

            return series;
        }

        /// <summary>A quiet CPU series at the scale the lab and the generator both show.</summary>
        private static double[] Base()
        {
            var rng = new Random(20260801);
            var series = new double[Samples];

            for (var i = 0; i < Samples; i++)
            {
                series[i] = 0.55 * (1.0 + ((rng.NextDouble() - 0.5) * 0.06));
            }

            return series;
        }

        private static int Env(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }
    }
}
