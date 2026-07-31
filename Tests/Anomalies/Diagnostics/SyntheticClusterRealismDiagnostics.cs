// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Is <see cref="SyntheticCluster"/> shaped like the real thing?
    ///
    /// <para><b>The false-positive rate is only worth what the generator is worth.</b> Every threshold decision
    /// in the guard leans on a number measured against synthetic pods, so the generator's own realism is
    /// load-bearing.</para>
    ///
    /// <para>Two spreads matter, and they pull the detector in opposite directions:</para>
    /// <list type="bullet">
    /// <item><b>Between pods.</b> How far apart replica medians sit. Drives the relative-gap gate directly:
    /// too wide and healthy replicas look like outliers.</item>
    /// <item><b>Within a pod.</b> How much one replica's own samples scatter. Drives Cliff's delta, which
    /// measures <i>overlap</i> — tight distributions barely overlap, so the same between-pod difference scores a
    /// far larger effect size. A generator with unrealistically quiet pods manufactures findings.</item>
    /// </list>
    ///
    /// <para><b>The lab side is computed from the recorded fixture, not quoted.</b> It used to be two arrays of
    /// constants transcribed by hand from one live run — which meant this diagnostic compared the generator
    /// against my own two-day-old retyping of the lab, could only speak about one metric, and could not notice
    /// if the lab changed. <see cref="LabWindowFixture"/> carries the faulted pod in its header, so the
    /// reference is the <i>healthy</i> replicas of a real window and every channel is covered.</para>
    ///
    /// <para><b>Two things had to be made comparable before any disagreement meant anything</b>, and both were
    /// wrong before:</para>
    /// <list type="bullet">
    /// <item><b>Equal sample counts.</b> <c>(max − min)</c> is a range, and a range grows with the number of
    /// samples for any distribution whatsoever. The lab window is 48 samples and the generator was read at 80,
    /// so part of the gap was arithmetic rather than a modelling error. The generator is now read over exactly
    /// the fixture's length.</item>
    /// <item><b>A robust spread beside the range.</b> Range over median is decided by the two most extreme
    /// scrapes in the window — precisely the scrape spikes both sides deliberately contain. The interquartile
    /// spread is reported next to it: where the two disagree, the range was measuring outliers, not scatter,
    /// and the detectors are rank-based for that exact reason.</item>
    /// </list>
    /// </summary>
    public sealed class SyntheticClusterRealismDiagnostics
    {
        /// <summary>Pods must clear this many finite samples to be characterised at all.</summary>
        private const int MinimumSamples = 8;

        /// <summary>Generator realisations averaged over, so a single random draw cannot pass for a shape.</summary>
        private const int Seeds = 16;

        private readonly ITestOutputHelper _output;

        public SyntheticClusterRealismDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void ComparesTheGeneratorsSpreadAgainstTheLab()
        {
            Assert.True(
                LabWindowFixture.Exists,
                $"the recorded lab window is missing from {LabWindowFixture.Path} — this diagnostic has no "
                + "reference without it, and the constants it used to fall back on were the problem");

            var (lab, faulted) = LabWindowFixture.Load();

            // The degraded replica is the thing being detected; including it in "how much do healthy replicas
            // differ" would calibrate the generator against the fault.
            var healthy = new List<int>();

            for (var p = 0; p < lab.Pods.Count; p++)
            {
                if (!faulted.Contains(lab.Pods[p]))
                {
                    healthy.Add(p);
                }
            }

            var report = new StringBuilder();

            report.Append($"LAB fixture: {lab.Pods.Count} pods ({healthy.Count} healthy, ")
                .Append($"{faulted.Count} faulted), {lab.Length} samples\n")
                .Append("generator read over the same window length, so the range figures are comparable\n\n");

            report.Append($"{"metric",-24}{"source",-10}{"pods",5}{"between",10}")
                .Append($"{"within/rng",12}{"within/iqr",12}\n");

            // Enough history that the generator is past its own warm-up, and many seeds because of what the
            // between-pod column actually is — see MedianOverSeeds.
            var three = Clusters(3);
            var twenty = Clusters(20);

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;

                var labProfile = FromLab(lab, healthy, metric);
                var threeProfile = MedianOverSeeds(three, metric, lab.Length);
                var twentyProfile = MedianOverSeeds(twenty, metric, lab.Length);

                if (!labProfile.HasValue && !threeProfile.HasValue)
                {
                    continue;
                }

                Row(report, metric.ToString(), "LAB", labProfile);
                Row(report, string.Empty, "GEN", threeProfile);
                Row(report, string.Empty, "GEN", twentyProfile);
                report.Append('\n');
            }

            _output.WriteLine(report.ToString());

            Assert.True(true, "reported, not asserted — the comparison is the product");
        }

        private static void Row(StringBuilder report, string metric, string source, Profile? profile)
        {
            if (profile is not { } p)
            {
                report.Append($"{metric,-24}{source,-10}{"—",5}{"absent",10}\n");

                return;
            }

            report.Append($"{metric,-24}{source,-10}{p.Pods,5}{p.Between,10:P1}")
                .Append($"{p.WithinRange,12:P1}{p.WithinIqr,12:P1}\n");
        }

        private static List<SyntheticCluster> Clusters(int pods)
        {
            var clusters = new List<SyntheticCluster>(Seeds);

            for (var s = 0; s < Seeds; s++)
            {
                clusters.Add(new SyntheticCluster(pods, hours: 6, scrapeSeconds: 15.0, seed: 20260729 + s));
            }

            return clusters;
        }

        /// <summary>
        /// The generator's profile at the median of many seeds.
        ///
        /// <para><b>Because one seed says almost nothing about the between-pod column.</b> That column is
        /// <c>(max − min)</c> over the pods' medians — a range over three draws when the group is three
        /// replicas, which is one of the noisiest statistics available. Tuning the per-pod offset against a
        /// single realisation of it is fitting to a coin flip: the same generator, unchanged, moved that
        /// number from 12% to 28% purely because an unrelated edit shifted the random stream.</para>
        ///
        /// <para>It does not rescue the <b>lab</b> side, which is three real pods and one realisation. That
        /// asymmetry is the honest limit of this comparison at the current lab size, and the within-pod
        /// columns — which average over dozens of samples per pod — are the ones to trust meanwhile.</para>
        /// </summary>
        private static Profile? MedianOverSeeds(List<SyntheticCluster> clusters, MetricIndex metric, int samples)
        {
            var between = new List<double>(clusters.Count);
            var ranges = new List<double>(clusters.Count);
            var iqrs = new List<double>(clusters.Count);
            var pods = 0;

            foreach (var cluster in clusters)
            {
                if (FromGenerator(cluster, metric, samples) is not { } profile)
                {
                    continue;
                }

                pods = profile.Pods;
                between.Add(profile.Between);
                ranges.Add(profile.WithinRange);
                iqrs.Add(profile.WithinIqr);
            }

            if (between.Count == 0)
            {
                return null;
            }

            between.Sort();
            ranges.Sort();
            iqrs.Sort();

            return new Profile(
                pods, Quantile(between, 0.5), Quantile(ranges, 0.5), Quantile(iqrs, 0.5));
        }

        private static Profile? FromLab(MetricWindow window, List<int> healthy, MetricIndex metric)
        {
            var series = new List<double[]>(healthy.Count);

            for (var i = 0; i < healthy.Count; i++)
            {
                series.Add(window.Series(healthy[i], metric).ToArray());
            }

            return Characterise(series);
        }

        private static Profile? FromGenerator(SyntheticCluster cluster, MetricIndex metric, int samples)
        {
            // The tail of the run: the generator's own start transient is not what the lab window contains.
            var start = Math.Max(0, cluster.Samples - samples - 1);
            var length = Math.Min(samples, cluster.Samples - start);
            var series = new List<double[]>();

            for (var p = 0; p < cluster.Pods; p++)
            {
                series.Add(cluster.Series(p, metric).AsSpan(start, length).ToArray());
            }

            return Characterise(series);
        }

        /// <summary>
        /// Between-pod spread of the medians, and the median pod's own scatter measured two ways. Null when
        /// too few pods carry usable data for either question to have an answer.
        /// </summary>
        private static Profile? Characterise(List<double[]> series)
        {
            var medians = new List<double>(series.Count);
            var ranges = new List<double>(series.Count);
            var iqrs = new List<double>(series.Count);

            foreach (var values in series)
            {
                var finite = new List<double>(values.Length);

                foreach (var value in values)
                {
                    if (double.IsFinite(value))
                    {
                        finite.Add(value);
                    }
                }

                if (finite.Count < MinimumSamples)
                {
                    continue;
                }

                finite.Sort();

                var median = Quantile(finite, 0.5);

                if (Math.Abs(median) <= 1e-12)
                {
                    // An identically-zero counter has no relative scale. It is not noise-free, it is
                    // unscaled — reporting 0% here would read as "the generator matches perfectly".
                    continue;
                }

                medians.Add(median);
                ranges.Add((finite[^1] - finite[0]) / median);
                iqrs.Add((Quantile(finite, 0.75) - Quantile(finite, 0.25)) / median);
            }

            if (medians.Count < 2)
            {
                return null;
            }

            medians.Sort();
            ranges.Sort();
            iqrs.Sort();

            var centre = Quantile(medians, 0.5);

            return new Profile(
                medians.Count,
                (medians[^1] - medians[0]) / centre,
                Quantile(ranges, 0.5),
                Quantile(iqrs, 0.5));
        }

        /// <summary>Nearest-rank quantile of an already sorted list.</summary>
        private static double Quantile(List<double> sorted, double q)
        {
            var index = (int)(q * (sorted.Count - 1));

            return sorted[Math.Clamp(index, 0, sorted.Count - 1)];
        }

        private readonly record struct Profile(int Pods, double Between, double WithinRange, double WithinIqr);
    }
}
