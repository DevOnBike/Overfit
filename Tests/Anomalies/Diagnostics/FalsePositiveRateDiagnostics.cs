// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Statistics;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// How many incidents the guard invents on a cluster where nothing is wrong.
    ///
    /// <para><b>This is the one number the guard has never had.</b> Every threshold in it was calibrated
    /// against detectability — can it find the replica we deliberately broke — and not one against false
    /// positives. A detector that finds every fault and also reports forty a day is worse than useless: it is
    /// the alert fatigue the product exists to reduce. The blueprint's M0 gate turns on this number, and
    /// reasoning cannot supply it.</para>
    ///
    /// <para><b>Needs no cluster.</b> <see cref="SyntheticCluster"/> generates the population, so this runs
    /// deterministically in the suite. That matters: the live lab can only ever show four pods for ten minutes,
    /// which is far too small a sample to say anything about a rate.</para>
    ///
    /// <para>Override the shape with <c>OVERFIT_FP_PODS</c>, <c>OVERFIT_FP_HOURS</c> and <c>OVERFIT_FP_SEED</c>.
    /// Several seeds are worth running: one population is one draw, and a rate read off a single draw is a
    /// point estimate with no spread.</para>
    /// </summary>
    public sealed class FalsePositiveRateDiagnostics
    {
        /// <summary>
        /// Detector window, in minutes. 20 min at a 15 s scrape is 80 samples — above every MinimumSamples
        /// floor — and is also the worst possible choice against a 24-hour cycle, which is why it is a knob:
        /// <c>OVERFIT_FP_WINDOW</c>.
        /// </summary>
        private static int WindowMinutes => Env("OVERFIT_FP_WINDOW", 20);

        /// <summary>How far the window advances between evaluations, i.e. the guard's cadence.</summary>
        private static int StepMinutes => Env("OVERFIT_FP_STEP", 5);

        /// <summary>Non-zero to test the seasonal residual instead of the raw series (<c>OVERFIT_FP_SEASONAL</c>).</summary>
        private static bool Seasonal => Env("OVERFIT_FP_SEASONAL", 0) != 0;

        /// <summary>Complete periods required before an expectation is built.</summary>
        private const int SeasonalPeriods = 2;

        private static readonly DateTimeOffset Origin = new(2026, 7, 29, 0, 0, 0, TimeSpan.Zero);

        private readonly ITestOutputHelper _output;

        public FalsePositiveRateDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void MeasuresTheFalsePositiveRateOnAHealthyPopulation()
        {
            var pods = Env("OVERFIT_FP_PODS", 20);
            var hours = Env("OVERFIT_FP_HOURS", 24);
            var seed = Env("OVERFIT_FP_SEED", 20260729);

            var cluster = new SyntheticCluster(pods, hours, scrapeSeconds: 15.0, seed: seed);

            var windowMinutes = WindowMinutes;
            var stepMinutes = StepMinutes;
            var windowSamples = (int)(windowMinutes * 60 / cluster.ScrapeSeconds);
            var stepSamples = (int)(stepMinutes * 60 / cluster.ScrapeSeconds);

            // Trend needs its own timestamps once per window shape, not once per pod per metric.
            var times = new double[windowSamples];
            for (var i = 0; i < windowSamples; i++)
            {
                times[i] = i * cluster.ScrapeSeconds;
            }

            var samplesPerPeriod = SeasonalBaseline.SamplesPerPeriod(
                TimeSpan.FromHours(24), TimeSpan.FromSeconds(cluster.ScrapeSeconds));

            // Both arms must evaluate the SAME windows, or the comparison is between two different populations
            // rather than two treatments. The seasonal arm cannot start before it has its history, so the raw
            // arm skips that stretch too.
            var warmup = SeasonalPeriods * samplesPerPeriod;
            var seasonal = Seasonal;
            var expectation = new double[windowSamples];
            var baselinesBuilt = 0;
            var baselinesMissing = 0;

            var peer = new PeerGroupOutlierDetector();
            var trend = new TrendDetector();

            // A single-node lab would relate every finding to every other through the node coordinate, and the
            // synthetic cluster is one deployment on unspecified nodes — so the node link carries nothing here
            // and is switched off rather than left to merge everything.
            var grouping = IncidentGroupingOptions.Balanced with
            {
                Topology = TopologyWeights.SingleNode
            };

            var incidents = 0;
            var findings = 0;
            var evaluations = 0;
            var bySignal = new SortedDictionary<string, int>(StringComparer.Ordinal);
            var byFamily = new SortedDictionary<string, int>(StringComparer.Ordinal);
            var accused = new HashSet<string>(StringComparer.Ordinal);

            var buffer = new PeerOutlierFinding[pods];
            var subjects = new IncidentSubject[pods];
            for (var p = 0; p < pods; p++)
            {
                subjects[p] = new IncidentSubject("overfit", "overfit-server", SyntheticCluster.PodName(p), string.Empty);
            }

            for (var start = warmup; start + windowSamples <= cluster.Samples; start += stepSamples)
            {
                evaluations++;

                var from = Origin.AddSeconds(start * cluster.ScrapeSeconds);
                var to = from.AddMinutes(windowMinutes);
                var pipeline = new IncidentPipeline();

                for (var m = 0; m < SyntheticCluster.MetricCount; m++)
                {
                    var metric = (MetricIndex)m;
                    var peers = new List<PeerSeries>(pods);

                    for (var p = 0; p < pods; p++)
                    {
                        var values = cluster.Series(p, metric).AsSpan(start, windowSamples).ToArray();
                        var work = IsLoadSensitive(metric)
                            ? cluster.Series(p, MetricIndex.RequestsPerSecond).AsSpan(start, windowSamples).ToArray()
                            : [];

                        peers.Add(new PeerSeries(SyntheticCluster.PodName(p), values, work));
                    }

                    var kind = IsLoadSensitive(metric) ? PeerSignalKind.LoadSensitive : PeerSignalKind.LoadIndependent;
                    var result = peer.Detect(peers, kind, PeerOutlierOptions.Balanced, buffer);

                    var added = pipeline.ObservePeerGroup(
                        metric.ToString(), result, buffer.AsSpan(0, pods), subjects.AsSpan(0, pods), from, to);

                    if (added > 0)
                    {
                        byFamily["peer"] = byFamily.GetValueOrDefault("peer") + added;
                    }

                    // Trend, per pod, over the same window.
                    for (var p = 0; p < pods; p++)
                    {
                        var history = cluster.Series(p, metric);
                        var values = history.AsSpan(start, windowSamples).ToArray();

                        var reference = ReadOnlySpan<double>.Empty;

                        if (seasonal)
                        {
                            if (SeasonalBaseline.TryBuild(
                                    history, start, windowSamples, samplesPerPeriod, expectation, SeasonalPeriods))
                            {
                                reference = expectation;
                                baselinesBuilt++;
                            }
                            else
                            {
                                baselinesMissing++;
                            }
                        }

                        var verdict = trend.Detect(values, times, TrendOptions.Balanced, double.NaN, reference);

                        if (pipeline.Observe(subjects[p], metric.ToString(), verdict, from, to))
                        {
                            byFamily["trend"] = byFamily.GetValueOrDefault("trend") + 1;
                        }
                    }
                }

                var grouped = pipeline.Group(grouping);

                incidents += grouped.Count;
                findings += pipeline.Count;

                for (var i = 0; i < grouped.Count; i++)
                {
                    foreach (var finding in grouped[i].Findings)
                    {
                        bySignal[finding.Signal] = bySignal.GetValueOrDefault(finding.Signal) + 1;
                        accused.Add(finding.Subject.Pod);
                    }
                }
            }

            var report = new StringBuilder();
            report.Append($"population   {pods} pods, {hours} h, seed {seed}\n");
            report.Append($"cadence      {WindowMinutes} min window every {StepMinutes} min = {evaluations} evaluations\n");
            report.Append($"\nincidents    {incidents}\n");
            report.Append($"findings     {findings}\n");
            report.Append($"pods accused {accused.Count} of {pods}\n");
            report.Append($"\nrate         {incidents / (double)hours:F2} incidents/hour  "
                          + $"({incidents * 24.0 / hours:F1} per day)\n");
            report.Append($"             {incidents / (double)evaluations:P1} of evaluations produced one\n");

            if (byFamily.Count > 0)
            {
                report.Append("\nby detector family\n");
                foreach (var (family, count) in byFamily)
                {
                    report.Append($"   {family,-8} {count}\n");
                }
            }

            if (bySignal.Count > 0)
            {
                report.Append("\nby signal — this is the actionable column: a rate concentrated in one metric is\n");
                report.Append("a calibration problem for that metric, not for the guard\n");
                foreach (var (signal, count) in bySignal)
                {
                    report.Append($"   {signal,-24} {count}\n");
                }
            }

            _output.WriteLine(report.ToString());

            // Reported, not asserted. The first run of a measurement has no baseline to fail against, and
            // picking a bound now would mean picking one that whatever came out happens to satisfy.
            Assert.True(evaluations > 0, "no window fitted in the generated history");
        }

        /// <summary>
        /// Raw magnitudes that scale with traffic, and therefore only comparable across peers once divided by a
        /// per-pod work metric. Fractions and per-request measures are already normalised.
        /// </summary>
        private static bool IsLoadSensitive(MetricIndex metric)
        {
            return metric is MetricIndex.CpuUsageRatio
                or MetricIndex.MemoryWorkingSetBytes
                or MetricIndex.GcGen2HeapBytes
                or MetricIndex.GcPauseRatio
                or MetricIndex.ThreadPoolQueueLength;
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
