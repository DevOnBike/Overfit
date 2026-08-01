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
    /// The shape of a peer group, member by member, for a signal the coherence gate refuses to judge.
    ///
    /// <para><b>Written before changing the gate, not after.</b> The detection matrix showed that a replica at
    /// 2.5× CPU is never named because the group verdict is <c>Inconclusive</c>, and the obvious repair — count
    /// only <i>material</i> departures instead of raw relative gaps — is a guess until the distribution is on
    /// the table. If the group's ordinary members are each individually significant against the rest, counting
    /// material departures changes nothing and the repair is the wrong one.</para>
    ///
    /// <para>So this prints every member's median, its gap from the group centre, whether the rank test calls
    /// it a deviation, and whether it clears the size gates — for one window, with and without an injected
    /// fault.</para>
    /// </summary>
    public sealed class PeerCoherenceDiagnostics
    {
        private const double ScrapeSeconds = 15.0;
        private const int WindowSamples = 80;
        private const int Target = 3;

        private readonly ITestOutputHelper _output;

        public PeerCoherenceDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void ShowsWhyTheGroupHasNoNorm()
        {
            var pods = Env("OVERFIT_COHERENCE_PODS", 12);
            var seed = Env("OVERFIT_COHERENCE_SEED", 20260801);
            var metric = Metric();

            var report = new StringBuilder();

            report.Append($"metric {metric}, {pods} pods, one {WindowSamples}-sample window, seed {seed}\n");

            Dump(report, "HEALTHY", pods, seed, metric, injectFactor: 1.0);
            Dump(report, "ONE POD AT 2.5x", pods, seed, metric, injectFactor: 2.5);

            _output.WriteLine(report.ToString());

            Assert.True(pods >= 3);
        }

        private static void Dump(
            StringBuilder report, string title, int pods, int seed, MetricIndex metric, double injectFactor)
        {
            var cluster = new SyntheticCluster(pods, 2.0, ScrapeSeconds, seed, restartsPerPodPerDay: 0.0);
            var start = cluster.Samples - WindowSamples;

            if (injectFactor != 1.0)
            {
                var series = cluster.Series(Target, metric);

                for (var i = 0; i < cluster.Samples; i++)
                {
                    series[i] *= injectFactor;
                }
            }

            var peers = new List<PeerSeries>(pods);
            var medians = new double[pods];

            for (var p = 0; p < pods; p++)
            {
                var values = cluster.Window(p, metric, start, WindowSamples).ToArray();

                // The same work series the guard supplies for a load-sensitive signal, so the numbers here are
                // the ones the detector actually compares rather than the raw metric.
                var work = PeerSignalCatalog.RequiresWork(metric)
                    ? cluster.Window(p, MetricIndex.RequestsPerSecond, start, WindowSamples).ToArray()
                    : [];

                peers.Add(new PeerSeries(SyntheticCluster.PodName(p), values, work));
                medians[p] = Median(values);
            }

            var kind = PeerSignalCatalog.Classify(metric);
            var options = PeerOutlierOptions.Balanced;
            var findings = new PeerOutlierFinding[pods];
            var result = new PeerGroupOutlierDetector().Detect(peers, kind, options, findings);

            var centre = Median(medians);

            report.Append($"\n=== {title} ===\n");
            report.Append($"group centre {centre:G4}   verdict {result.Status}   "
                          + $"high {result.HighCount} low {result.LowCount} comparable {result.PeerCount}\n");
            report.Append($"   {result.Reason}\n\n");

            report.Append($"   {"pod",5}{"median",12}{"gap vs centre",16}{"rank says",12}"
                          + $"{"material",10}{"delta",8}\n");

            var raw = 0;
            var material = 0;

            for (var p = 0; p < pods; p++)
            {
                var gap = Math.Abs(centre) > 1e-12 ? Math.Abs(medians[p] - centre) / Math.Abs(centre) : 0.0;
                var isRaw = gap >= options.MinRelativeGap;
                var isMaterial = findings[p].Deviation != PeerDeviation.None;

                raw += isRaw ? 1 : 0;
                material += isMaterial ? 1 : 0;

                report.Append($"   {p,5}{medians[p],12:G4}{gap,15:P0}"
                              + $"{(isRaw ? " over 8%" : " -"),12}"
                              + $"{(isMaterial ? "YES" : "-"),10}"
                              + $"{Math.Abs(findings[p].Comparison.EffectSize),8:F2}\n");
            }

            report.Append($"\n   raw departures (gap >= 8%): {raw} of {pods}"
                          + $"   -> coherence gate trips at {(pods / 3) + 1}\n");
            report.Append($"   material departures:        {material} of {pods}\n");
        }

        private static double Median(double[] values)
        {
            var copy = (double[])values.Clone();

            Array.Sort(copy);

            return copy[copy.Length / 2];
        }

        private static MetricIndex Metric()
        {
            var raw = Environment.GetEnvironmentVariable("OVERFIT_COHERENCE_METRIC");

            return Enum.TryParse<MetricIndex>(raw, ignoreCase: true, out var metric)
                ? metric
                : MetricIndex.CpuUsageRatio;
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
