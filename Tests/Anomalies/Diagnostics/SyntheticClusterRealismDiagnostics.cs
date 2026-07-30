// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using System.Text;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Is <see cref="SyntheticCluster"/> shaped like the real thing?
    ///
    /// <para><b>The false-positive rate is only worth what the generator is worth.</b> Every threshold decision
    /// in the guard now leans on a number measured against synthetic pods, so the generator's own realism is
    /// load-bearing — and it has never been checked against the cluster lab it was supposedly built from.</para>
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
    /// <para>Reference values are the ones measured on the lab's three healthy replicas over a loaded
    /// twelve-minute window, recorded here so the comparison is against data rather than intuition.</para>
    /// </summary>
    public sealed class SyntheticClusterRealismDiagnostics
    {
        /// <summary>p95 medians of the lab's three healthy replicas, in milliseconds.</summary>
        private static readonly double[] LabP95Medians = [860.42, 880.36, 901.52];

        /// <summary>Their within-pod spreads, <c>(max − min) / median</c>, as measured.</summary>
        private static readonly double[] LabP95WithinSpread = [0.518, 0.547, 0.223];

        private readonly ITestOutputHelper _output;

        public SyntheticClusterRealismDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void ComparesTheGeneratorsSpreadAgainstTheLab()
        {
            const int WindowSamples = 80;   // 20 min at 15 s, the detector's window

            var report = new StringBuilder();

            report.Append("LAB — three healthy replicas, loaded 12-minute window\n");
            report.Append($"   p95 medians          {string.Join(" / ", LabP95Medians)} ms\n");
            report.Append($"   between-pod spread   {Spread(LabP95Medians):P1}\n");
            report.Append($"   within-pod spread    {string.Join(" / ", Array.ConvertAll(LabP95WithinSpread, s => s.ToString("P0")))}\n");
            report.Append($"   median within-pod    {Median(new List<double>(LabP95WithinSpread)):P0}\n\n");

            foreach (var pods in new[] { 3, 20 })
            {
                var cluster = new SyntheticCluster(pods, hours: 6, scrapeSeconds: 15.0, seed: 20260729);
                var start = cluster.Samples - WindowSamples - 1;

                var medians = new List<double>(pods);
                var within = new List<double>(pods);

                for (var p = 0; p < pods; p++)
                {
                    var window = cluster.Series(p, MetricIndex.LatencyP95Ms).AsSpan(start, WindowSamples);
                    var finite = new List<double>(WindowSamples);

                    for (var i = 0; i < window.Length; i++)
                    {
                        if (double.IsFinite(window[i]))
                        {
                            finite.Add(window[i]);
                        }
                    }

                    if (finite.Count == 0)
                    {
                        continue;
                    }

                    var median = Median(finite);
                    medians.Add(median);

                    finite.Sort();
                    within.Add(median > 0.0 ? (finite[^1] - finite[0]) / median : double.NaN);
                }

                medians.Sort();

                report.Append($"GENERATOR — {pods} pods\n");
                report.Append($"   p95 medians          {medians[0]:F0} … {Median(medians):F0} … {medians[^1]:F0} ms\n");
                report.Append($"   between-pod spread   {Spread(medians.ToArray()):P1}\n");
                report.Append($"   median within-pod    {Median(within):P0}\n\n");
            }

            _output.WriteLine(report.ToString());

            Assert.True(true, "reported, not asserted — the comparison is the product");
        }

        /// <summary><c>(max − min) / median</c>, the same figure quoted for the lab.</summary>
        private static double Spread(double[] values)
        {
            var copy = new List<double>(values);
            copy.Sort();

            var median = Median(copy);

            return median > 0.0 ? (copy[^1] - copy[0]) / median : double.NaN;
        }

        private static double Median(List<double> values)
        {
            var finite = new List<double>(values.Count);

            foreach (var value in values)
            {
                if (double.IsFinite(value))
                {
                    finite.Add(value);
                }
            }

            if (finite.Count == 0)
            {
                return double.NaN;
            }

            finite.Sort();

            return finite[finite.Count / 2];
        }
    }
}
