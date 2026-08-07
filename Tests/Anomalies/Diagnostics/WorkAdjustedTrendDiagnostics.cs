// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Three ways to stop the daily traffic curve from looking like a CPU trend, scored against each other.
    ///
    /// <para><b>The problem is measured, on the live lab.</b> The dominant false-positive source there is the
    /// trend family on <c>CpuUsageRatio</c>, and the findings are mostly <i>downward</i> — "Series fell by
    /// 12.5% of typical, tau -0.47, lag-1 autocorrelation 0.80". CPU follows traffic and traffic follows the
    /// time of day, so a twenty-minute window on the slope of the daily curve finds a real and entirely
    /// meaningless drift.</para>
    ///
    /// <para><b>And the obvious repair is a trap this codebase has already documented in the other
    /// dimension.</b> Dividing by the work metric is right only when the signal is proportional to work.
    /// Measured, and the generator is built on it: CPU here is <b>affine</b> — <c>0.45 + 0.040 × traffic</c> —
    /// so the fixed floor is 22% of the value at peak and 58% at the trough, and <c>cpu / traffic</c> swings
    /// almost twofold across the day in the <i>opposite</i> direction to traffic. Naive division does not
    /// remove the daily curve; it inverts it.</para>
    ///
    /// <para><b>So the third treatment is the affine one the peer catalog names as the general answer:</b> fit
    /// <c>value = fixed + marginal × work</c> robustly over the window and ask what the value would have been
    /// at constant work. That subtracts the traffic-driven component whatever the mix of fixed and marginal
    /// cost, and — unlike a ratio — leaves the series in its own units and at its own level, so the relative
    /// gate downstream still means something.</para>
    ///
    /// <para><b>Both columns are scored, because either one alone can be satisfied by a treatment that is
    /// useless.</b> A series flattened hard enough reports nothing at all, so the same three treatments are
    /// also run against a replica whose per-request CPU cost genuinely regresses.</para>
    /// </summary>
    public sealed class WorkAdjustedTrendDiagnostics
    {
        private const double ScrapeSeconds = 15.0;

        private readonly ITestOutputHelper _output;

        public WorkAdjustedTrendDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact("1s")]
        public void ComparesRawDividedAndAffineAdjustedTrends()
        {
            var pods = Env("OVERFIT_ADJ_PODS", 12);
            var hours = Env("OVERFIT_ADJ_HOURS", 24);
            var seed = Env("OVERFIT_ADJ_SEED", 20260801);
            var windowMinutes = Env("OVERFIT_ADJ_WINDOW", 20);
            var stepMinutes = Env("OVERFIT_ADJ_STEP", 5);

            var report = new StringBuilder();

            report.Append($"{pods} pods, {hours} h, seed {seed}, {windowMinutes} min window every "
                          + $"{stepMinutes} min\n");
            report.Append("CpuUsageRatio only — the signal the lab reports on, and the one that follows "
                          + "traffic\n\n");

            var healthy = new SyntheticCluster(pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);
            var regressed = new SyntheticCluster(pods, hours, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);

            // A per-request cost regression on one replica: the marginal term rises by 40% half-way through,
            // which is what a bad deployment of one canary looks like. Deliberately NOT a step in the fixed
            // term — that would be caught by anything, and the question here is whether the adjustment keeps
            // the case it is most at risk of erasing.
            Regress(regressed, pod: 3, factor: 1.4);

            report.Append($"   {"treatment",-22}{"false trends",14}{"per day",10}"
                          + $"{"   regression still found in",-30}\n");

            foreach (var treatment in new[] { Treatment.Raw, Treatment.Divided, Treatment.AffineAdjusted })
            {
                var noise = Count(healthy, pods, windowMinutes, stepMinutes, treatment, target: -1);
                var caught = Count(regressed, pods, windowMinutes, stepMinutes, treatment, target: 3);

                report.Append($"   {Name(treatment),-22}{noise,14}{noise * 24.0 / hours,10:F0}"
                              + $"   {caught,4} window(s)\n");
            }

            report.Append("\nfalse trends  Anomalous trend verdicts on a population where nothing is wrong\n");
            report.Append("regression    windows in which the regressed replica was called anomalous\n");
            report.Append("A treatment that is quiet in BOTH columns has not fixed anything — it has gone deaf.\n");

            _output.WriteLine(report.ToString());

            Assert.True(pods >= 3);
        }

        private enum Treatment
        {
            Raw,
            Divided,
            AffineAdjusted,
        }

        private static string Name(Treatment treatment)
        {
            return treatment switch
            {
                Treatment.Raw => "raw (today)",
                Treatment.Divided => "divided by work",
                _ => "affine-adjusted",
            };
        }

        /// <summary>
        /// Anomalous trend verdicts across the whole history. With <paramref name="target"/> below zero every
        /// pod is counted; otherwise only that one, which is how the second column is measured.
        /// </summary>
        private static int Count(
            SyntheticCluster cluster, int pods, int windowMinutes, int stepMinutes,
            Treatment treatment, int target)
        {
            var samples = (int)(windowMinutes * 60 / ScrapeSeconds);
            var step = (int)(stepMinutes * 60 / ScrapeSeconds);
            var times = new double[samples];

            for (var i = 0; i < samples; i++)
            {
                times[i] = i * ScrapeSeconds;
            }

            var detector = new TrendDetector();
            var series = new double[samples];
            var work = new double[samples];
            var found = 0;

            for (var start = 0; start + samples <= cluster.Samples; start += step)
            {
                for (var pod = 0; pod < pods; pod++)
                {
                    if (target >= 0 && pod != target)
                    {
                        continue;
                    }

                    cluster.Window(pod, MetricIndex.CpuUsageRatio, start, samples).CopyTo(series);
                    cluster.Window(pod, MetricIndex.RequestsPerSecond, start, samples).CopyTo(work);

                    var judged = Apply(treatment, series, work);
                    var verdict = detector.Detect(judged, times, TrendOptions.Balanced);

                    found += verdict.Status == DetectionStatus.Anomalous ? 1 : 0;
                }
            }

            return found;
        }

        private static double[] Apply(Treatment treatment, double[] series, double[] work)
        {
            if (treatment == Treatment.Raw)
            {
                return (double[])series.Clone();
            }

            var result = new double[series.Length];

            if (treatment == Treatment.Divided)
            {
                for (var i = 0; i < series.Length; i++)
                {
                    result[i] = Math.Abs(work[i]) > 1e-12 ? series[i] / work[i] : double.NaN;
                }

                return result;
            }

            // What the signal would have been at constant work: value − marginal × (work − typical work).
            // The level and the units survive, which a ratio destroys and which every gate downstream needs.
            var marginal = MarginalCost(series, work);
            var typical = Median(work);

            for (var i = 0; i < series.Length; i++)
            {
                result[i] = series[i] - (marginal * (work[i] - typical));
            }

            return result;
        }

        /// <summary>
        /// Theil-Sen slope of value against work — the marginal cost per unit of work, fitted robustly so a
        /// handful of bursts cannot set it.
        /// </summary>
        private static double MarginalCost(double[] series, double[] work)
        {
            var slopes = new List<double>(series.Length * 2);

            // Every pair is O(n²) and n is 80 here, which is 3160 pairs — fine for a diagnostic. A production
            // implementation would subsample, and that is a decision to make against a measurement.
            for (var i = 0; i < series.Length; i++)
            {
                for (var j = i + 1; j < series.Length; j++)
                {
                    var dw = work[j] - work[i];

                    if (Math.Abs(dw) > 1e-9 && double.IsFinite(series[i]) && double.IsFinite(series[j]))
                    {
                        slopes.Add((series[j] - series[i]) / dw);
                    }
                }
            }

            if (slopes.Count == 0)
            {
                return 0.0;
            }

            slopes.Sort();

            return slopes[slopes.Count / 2];
        }

        /// <summary>Raises one replica's marginal cost per request, half-way through the history.</summary>
        private static void Regress(SyntheticCluster cluster, int pod, double factor)
        {
            var cpu = cluster.Series(pod, MetricIndex.CpuUsageRatio);
            var rps = cluster.Series(pod, MetricIndex.RequestsPerSecond);
            var from = cluster.Samples / 2;

            for (var i = from; i < cluster.Samples; i++)
            {
                // The generator's own model: 0.45 fixed plus 0.040 per request. Only the marginal term moves,
                // so the extra cost is proportional to what the replica is actually serving.
                cpu[i] += 0.040 * rps[i] * (factor - 1.0);
            }
        }

        private static double Median(double[] values)
        {
            var copy = (double[])values.Clone();

            Array.Sort(copy);

            return copy[copy.Length / 2];
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
