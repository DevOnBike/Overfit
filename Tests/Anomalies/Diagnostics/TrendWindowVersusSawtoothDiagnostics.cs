// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// Why the trend detector fires on a healthy heap, and which window length stops it.
    ///
    /// <para><b>The shadow run's only false positives were trend findings, and most were memory or heap.</b>
    /// Every one read the same way: "rose by 10–12% of typical across 15 minutes", tau 0.57–0.90. Those
    /// numbers are not noise — they are the GC sawtooth, whose amplitude was calibrated at <b>8.7%</b> and
    /// whose period is <b>57 scrapes</b>, which at a 15-second scrape is 14 minutes. The trend window is 15.
    /// A window the length of one period lands on the rising limb and sees a clean monotone climb, because
    /// over that stretch it <i>is</i> one.</para>
    ///
    /// <para>This is a property of any cyclic signal measured over one period, not a memory bug, which is why
    /// the fix has to be chosen by measurement rather than by argument. Two candidates, opposite costs:</para>
    /// <list type="bullet">
    /// <item><b>A longer window.</b> Several periods average the cycle out. Costs detection latency on a real
    /// leak and more data per cycle.</item>
    /// <item><b>Drop the trend family on these channels.</b> Free, and blinds the guard to exactly the fault
    /// it exists for.</item>
    /// </list>
    ///
    /// <para>So both arms are measured against both things that matter: false positives on a population where
    /// nothing is wrong, and detection of a leak that genuinely is there.</para>
    /// </summary>
    public sealed class TrendWindowVersusSawtoothDiagnostics
    {
        /// <summary>Scrapes per sawtooth cycle in the calibrated generator — see <c>MemoryCycleSamples</c>.</summary>
        private const int SawtoothPeriod = 92;

        private const int Pods = 20;

        private readonly ITestOutputHelper _output;

        public TrendWindowVersusSawtoothDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void MeasuresFalsePositivesAndLeakDetectionAgainstWindowLength()
        {
            var report = new StringBuilder();

            report.Append($"reclaim cycle {SawtoothPeriod} scrapes (23 min at 15 s), "
                          + $"amplitude {SyntheticCluster.MemorySawtoothAmplitude:P1}\n");
            report.Append($"{Pods} healthy pods, no fault injected — every finding below is a false positive\n\n");

            report.Append($"{"window",10}{"periods",9}{"mem FP",9}{"heap FP",9}{"leak found",12}\n");

            foreach (var multiple in new[] { 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 })
            {
                var length = (int)(SawtoothPeriod * multiple);
                var cluster = new SyntheticCluster(Pods, hours: 12, scrapeSeconds: 15.0, seed: 20260731);

                var memory = CountFindings(cluster, MetricIndex.MemoryWorkingSetBytes, length, 0.0);
                var heap = CountFindings(cluster, MetricIndex.GcGen2HeapBytes, length, 0.0);
                var leak = LeakIsFound(cluster, length, 0.0);

                report.Append($"{length,10}{multiple,9:F1}{memory,9}{heap,9}{(leak ? "yes" : "NO"),12}\n");
            }

            // The other lever, and the one that does not cost detection latency: a floor in bytes. The
            // window stays at the detector's default; only the absolute gate moves.
            var windowLength = (int)(SawtoothPeriod * 1.0);

            report.Append($"\n\nabsolute floor sweep at a {windowLength}-scrape window "
                          + "(the configuration the shadow run used)\n\n");
            report.Append($"{"floor MB",10}{"mem FP",9}{"heap FP",9}{"leak found",12}\n");

            foreach (var megabytes in new[] { 0, 25, 50, 100, 150, 200, 300, 500 })
            {
                var floor = megabytes * 1e6;
                var cluster = new SyntheticCluster(Pods, hours: 12, scrapeSeconds: 15.0, seed: 20260731);

                var memory = CountFindings(cluster, MetricIndex.MemoryWorkingSetBytes, windowLength, floor);
                var heap = CountFindings(cluster, MetricIndex.GcGen2HeapBytes, windowLength, floor);
                var leak = LeakIsFound(cluster, windowLength, floor);

                report.Append($"{megabytes,10}{memory,9}{heap,9}{(leak ? "yes" : "NO"),12}\n");
            }

            report.Append("\nmem FP / heap FP: pods out of ")
                .Append(Pods)
                .Append(" reported as trending, on a population where nothing is wrong.\n");
            report.Append("leak found: a 40% rise injected over the window on one pod's working set — the "
                          + "fault this detector exists for.\n");

            _output.WriteLine(report.ToString());

            Assert.True(true, "reported, not asserted — the table is the product");
        }

        private static int CountFindings(
            SyntheticCluster cluster, MetricIndex metric, int length, double floor = 0.0)
        {
            var start = cluster.Samples - length - 1;
            var findings = 0;

            for (var pod = 0; pod < cluster.Pods; pod++)
            {
                var series = cluster.Series(pod, metric).AsSpan(start, length).ToArray();

                if (Evaluate(series, floor).Status == DetectionStatus.Anomalous)
                {
                    findings++;
                }
            }

            return findings;
        }

        /// <summary>
        /// The other half of the trade. A window long enough to hide the sawtooth must still catch a leak, or
        /// the "fix" is just a way of reporting nothing.
        /// </summary>
        private static bool LeakIsFound(SyntheticCluster cluster, int length, double floor = 0.0)
        {
            var start = cluster.Samples - length - 1;
            var series = cluster.Series(0, MetricIndex.MemoryWorkingSetBytes).AsSpan(start, length).ToArray();
            var baseline = series[0];

            // A monotone 40% climb laid on top of the pod's own sawtooth: what a leak actually looks like,
            // rather than a clean ramp the detector would find trivially.
            for (var i = 0; i < series.Length; i++)
            {
                series[i] += baseline * 0.40 * i / (series.Length - 1);
            }

            return Evaluate(series, floor).Status == DetectionStatus.Anomalous;
        }

        private static TrendResult Evaluate(double[] series, double floor)
        {
            var times = new double[series.Length];

            for (var i = 0; i < times.Length; i++)
            {
                times[i] = i * 15.0;
            }

            var options = TrendOptions.Balanced with
            {
                MinAbsoluteChangeOverWindow = floor
            };

            return new TrendDetector().Detect(series, times, options);
        }
    }
}
