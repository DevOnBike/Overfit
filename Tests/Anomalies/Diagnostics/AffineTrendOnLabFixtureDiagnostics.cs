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
                          + $"{"work-prop.",13}{"fixed step",13}\n");

            foreach (var metric in new[] { MetricIndex.CpuUsageRatio, MetricIndex.GcPauseRatio })
            {
                // Built once per signal: both are the same replica, the same start point and — by
                // construction — the same MEAN effect. Only their relationship to work differs.
                var proportional = Regressed(recorded, metric, pod: 3, factor: 1.5, FaultShape.WorkProportional);
                var fixedStep = Regressed(recorded, metric, pod: 3, factor: 1.5, FaultShape.FixedStep);

                foreach (var treatment in new[] { Treatment.Raw, Treatment.Divided, Treatment.Affine })
                {
                    var healthy = Count(recorded, metric, windowMinutes, stepMinutes, treatment, -1, out var total);
                    var caughtProportional = Count(
                        proportional, metric, windowMinutes, stepMinutes, treatment, 3, out _);
                    var caughtFixed = Count(
                        fixedStep, metric, windowMinutes, stepMinutes, treatment, 3, out _);

                    report.Append($"   {metric,-22}{Name(treatment),-20}{healthy,14}{total,12}"
                                  + $"{caughtProportional,10} win.{caughtFixed,8} win.\n");
                }

                report.Append('\n');
            }

            report.Append("false trends  Anomalous trend verdicts on the RECORDED data, where nothing was wrong\n");
            report.Append("work-prop.    windows catching a fault PROPORTIONAL to work — the only shape scored\n");
            report.Append("              until 2026-08-09, and the reason the earlier comparison was uninformative\n");
            report.Append("fixed step    windows catching a fault INDEPENDENT of work, at the same mean size\n");
            report.Append("A treatment quiet in BOTH fault columns has not fixed anything — it has gone deaf.\n");
            report.Append("\nWhy the second fault column exists. Division and the affine fit differ ONLY in how\n");
            report.Append("they treat a fixed cost: dividing by work spreads a constant step across a varying\n");
            report.Append("denominator, while the affine fit subtracts only the work-driven part and leaves it.\n");
            report.Append("A fault built as marginal x work is invisible to that distinction — both treatments\n");
            report.Append("remove work-proportional structure — so a comparison scored on it alone CANNOT\n");
            report.Append("separate them, and the 11/11, 4/4 tie recorded in ROADMAP.md was a property of the\n");
            report.Append("experiment rather than a finding about the treatments. A longer recording would have\n");
            report.Append("reproduced that tie however long it ran.\n");

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

        /// <summary>Which way the injected fault relates to what the replica is serving.</summary>
        private enum FaultShape
        {
            /// <summary>
            /// Per-request cost rises — the shape of a bad canary. <b>The only shape scored until
            /// 2026-08-09</b>, and the reason the comparison it fed could not distinguish its arms.
            /// </summary>
            WorkProportional,

            /// <summary>
            /// A constant added regardless of load — a leaked buffer, a background loop, a sidecar. This is
            /// the <b>only regime in which the affine fit and plain division differ</b>: division spreads a
            /// constant across a varying denominator and inflates it exactly when traffic falls, while the
            /// affine fit subtracts only the work-driven component and leaves the step standing.
            /// </summary>
            FixedStep,
        }

        /// <summary>
        /// A copy of the recording with one replica degraded half-way through, in one of two shapes.
        ///
        /// <para><b>Both shapes are sized to the same mean effect</b>, so the two fault columns differ in
        /// their relationship to work and in nothing else. Without that the comparison would confound the
        /// shape of the fault with its size, and a treatment could win a column by being scored against a
        /// larger fault.</para>
        /// </summary>
        private static MetricWindow Regressed(
            MetricWindow source, MetricIndex metric, int pod, double factor, FaultShape shape)
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

            if (shape == FaultShape.WorkProportional)
            {
                for (var i = from; i < copy.Length; i++)
                {
                    target[i] += marginal * work[i] * (factor - 1.0);
                }

                return copy;
            }

            // The same total cost, delivered flat. Sized from the mean work over the degraded half, which is
            // what the proportional arm adds on average across the same samples.
            var meanWork = 0.0;
            var counted = 0;

            for (var i = from; i < copy.Length; i++)
            {
                if (double.IsFinite(work[i]))
                {
                    meanWork += work[i];
                    counted++;
                }
            }

            var step = counted > 0 ? marginal * (meanWork / counted) * (factor - 1.0) : 0.0;

            for (var i = from; i < copy.Length; i++)
            {
                target[i] += step;
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
