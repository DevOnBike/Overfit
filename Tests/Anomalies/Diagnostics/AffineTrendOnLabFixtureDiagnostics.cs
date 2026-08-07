// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Three ways to stop the daily traffic curve looking like a CPU trend, scored on <b>recorded cluster
    /// data</b> rather than on a generator.
    ///
    /// <para><b>The first attempt at this measurement was vacuous and that is why the recording exists.</b>
    /// Run against <see cref="SyntheticCluster"/>, every treatment scored zero false trends — the generator
    /// simply does not contain the phenomenon. Its diurnal curve moves traffic about 3% across a twenty-minute
    /// window against a gate that needs 10%, while the real lab drifts past that gate in roughly one window in
    /// ten. A comparison where no arm can score is not evidence about any arm.</para>
    ///
    /// <para><b>What is being tested.</b> Peer comparison divides a load-sensitive signal by a work metric;
    /// the trend family does not, so it follows traffic. Plain division is the obvious repair and is right only
    /// where cost is proportional to work — measured, CPU is <b>affine</b>, a fixed floor plus a marginal cost,
    /// and dividing by work then inflates the quotient exactly when traffic falls. The third treatment fits
    /// <c>value = fixed + marginal × work</c> over the window and asks what the value would have been at
    /// constant work, which subtracts the traffic-driven component whatever the mix and leaves the series in
    /// its own units.</para>
    ///
    /// <para><b>Both columns are scored.</b> A treatment that reports nothing has not removed the daily curve,
    /// it has removed the detector — so the same three run against a copy of the recording with a genuine
    /// per-request cost regression injected into one replica.</para>
    ///
    /// <para>Needs <c>lab-window-healthy-12pod.csv</c>. Knobs: <c>OVERFIT_AFFINE_FIXTURE</c>,
    /// <c>OVERFIT_AFFINE_WINDOW</c> (minutes), <c>OVERFIT_AFFINE_STEP</c> (minutes).</para>
    /// </summary>
    public sealed class AffineTrendOnLabFixtureDiagnostics
    {
        private readonly ITestOutputHelper _output;

        public AffineTrendOnLabFixtureDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact("149ms")]
        public void ComparesRawDividedAndAffineOnRecordedClusterData()
        {
            var name = Environment.GetEnvironmentVariable("OVERFIT_AFFINE_FIXTURE")
                       ?? "lab-window-healthy-12pod.csv";

            Environment.SetEnvironmentVariable("OVERFIT_LAB_FIXTURE_NAME", name);

            Assert.True(LabWindowFixture.Exists, $"no recording at {LabWindowFixture.Path}");

            var (recorded, _) = LabWindowFixture.Load();
            var windowMinutes = Setting("OVERFIT_AFFINE_WINDOW", 20);
            var stepMinutes = Setting("OVERFIT_AFFINE_STEP", 5);

            var report = new StringBuilder();

            report.Append($"recording {name}\n");
            report.Append($"{recorded.Pods.Count} pods, {recorded.Length} scrapes at "
                          + $"{recorded.Step.TotalSeconds:F0} s, {windowMinutes} min window every "
                          + $"{stepMinutes} min\n\n");

            report.Append($"   {"signal",-22}{"treatment",-20}{"false trends",14}{"of windows",12}"
                          + $"{"regression seen in",20}\n");

            foreach (var metric in new[] { MetricIndex.CpuUsageRatio, MetricIndex.GcPauseRatio })
            {
                foreach (var treatment in new[] { Treatment.Raw, Treatment.Divided, Treatment.Affine })
                {
                    var healthy = Count(recorded, metric, windowMinutes, stepMinutes, treatment, -1, out var total);
                    var broken = Count(
                        Regressed(recorded, metric, pod: 3, factor: 1.5),
                        metric, windowMinutes, stepMinutes, treatment, 3, out _);

                    report.Append($"   {metric,-22}{Name(treatment),-20}{healthy,14}{total,12}"
                                  + $"{broken,17} win.\n");
                }

                report.Append('\n');
            }

            report.Append("false trends  Anomalous trend verdicts on the RECORDED data, where nothing was wrong\n");
            report.Append("regression    windows in which the deliberately regressed replica was called anomalous\n");
            report.Append("A treatment quiet in BOTH columns has not fixed anything — it has gone deaf.\n");

            _output.WriteLine(report.ToString());

            Assert.True(recorded.Pods.Count >= 3);
        }

        private enum Treatment
        {
            Raw,
            Divided,
            Affine,
        }

        private static string Name(Treatment treatment)
            => treatment switch
            {
                Treatment.Raw => "raw (today)",
                Treatment.Divided => "divided by work",
                _ => "affine-adjusted",
            };

        /// <summary>
        /// Anomalous trend verdicts across every sliding window. <paramref name="target"/> below zero counts
        /// every pod; otherwise only that one.
        /// </summary>
        private static int Count(
            MetricWindow source, MetricIndex metric, int windowMinutes, int stepMinutes,
            Treatment treatment, int target, out int windows)
        {
            var samples = (int)(windowMinutes * 60 / source.Step.TotalSeconds);
            var step = (int)(stepMinutes * 60 / source.Step.TotalSeconds);
            var times = new double[samples];

            for (var i = 0; i < samples; i++)
            {
                times[i] = i * source.Step.TotalSeconds;
            }

            var detector = new TrendDetector();
            var series = new double[samples];
            var work = new double[samples];
            var found = 0;

            windows = 0;

            for (var start = 0; start + samples <= source.Length; start += step)
            {
                windows++;

                for (var pod = 0; pod < source.Pods.Count; pod++)
                {
                    if (target >= 0 && pod != target)
                    {
                        continue;
                    }

                    source.Series(pod, metric).Slice(start, samples).CopyTo(series);
                    source.Series(pod, MetricIndex.RequestsPerSecond).Slice(start, samples).CopyTo(work);

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

            // What the signal would have been at constant work. The level and the units survive, which a
            // ratio destroys and which every gate downstream needs.
            var marginal = MarginalCost(series, work);
            var typical = Median(work);

            for (var i = 0; i < series.Length; i++)
            {
                result[i] = series[i] - (marginal * (work[i] - typical));
            }

            return result;
        }

        /// <summary>Theil-Sen slope of value against work — the marginal cost, fitted robustly.</summary>
        private static double MarginalCost(double[] series, double[] work)
        {
            var slopes = new List<double>(series.Length * 4);

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

        /// <summary>
        /// A copy of the recording with one replica's per-request cost raised half-way through — the shape of
        /// a bad canary, and deliberately proportional to what that replica is serving, so a treatment that
        /// removes the traffic component is at maximum risk of erasing it.
        /// </summary>
        private static MetricWindow Regressed(MetricWindow source, MetricIndex metric, int pod, double factor)
        {
            var copy = new MetricWindow(source.Pods, source.Length, source.Start, source.Step);

            for (var p = 0; p < source.Pods.Count; p++)
            {
                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    source.Series(p, (MetricIndex)m).CopyTo(copy.Series(p, (MetricIndex)m));
                }
            }

            var target = copy.Series(pod, metric);
            var work = copy.Series(pod, MetricIndex.RequestsPerSecond);
            var from = source.Length / 2;
            var marginal = MarginalCost(source.Series(pod, metric).ToArray(), work.ToArray());

            for (var i = from; i < copy.Length; i++)
            {
                target[i] += marginal * work[i] * (factor - 1.0);
            }

            return copy;
        }

        private static double Median(double[] values)
        {
            var copy = (double[])values.Clone();

            Array.Sort(copy);

            return copy[copy.Length / 2];
        }

        private static int Setting(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }
    }
}
