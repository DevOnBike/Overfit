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
using DevOnBike.Overfit.Anomalies.Monitoring;
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

        /// <summary>
        /// Comma-separated <see cref="MetricIndex"/> names excluded from <b>trend</b> testing
        /// (<c>OVERFIT_FP_TREND_SKIP</c>); the peer comparison still sees them.
        ///
        /// <para>An ablation knob, not a feature. The breakdown by signal attributed roughly 70% of findings to
        /// memory and GC heap, but findings are not incidents — the grouper merges them — so removing 70% of
        /// findings need not remove 70% of incidents. This measures which it is before anything is built.</para>
        /// </summary>
        private static HashSet<MetricIndex> TrendSkip()
        {
            var skip = new HashSet<MetricIndex>();
            var raw = Environment.GetEnvironmentVariable("OVERFIT_FP_TREND_SKIP");

            if (string.IsNullOrWhiteSpace(raw))
            {
                return skip;
            }

            foreach (var name in raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (!Enum.TryParse<MetricIndex>(name, ignoreCase: true, out var metric))
                {
                    throw new ArgumentException($"OVERFIT_FP_TREND_SKIP names an unknown metric: '{name}'.");
                }

                skip.Add(metric);
            }

            return skip;
        }

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

            // OVERFIT_FP_RESTARTS=0 ablates restarts. A restarted pod's memory ramp and the GC sawtooth are two
            // different mechanisms producing the same-looking finding, and turning one off is the only way to
            // say which one a count comes from.
            var restarts = Env("OVERFIT_FP_RESTARTS", 1) != 0 ? 1.0 : 0.0;
            var cluster = new SyntheticCluster(
                pods, hours, scrapeSeconds: 15.0, seed: seed, restartsPerPodPerDay: restarts);

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
            var trendSkip = TrendSkip();
            var absoluteGate = Env("OVERFIT_FP_ABSOLUTE", 1) != 0;

            // Both floor levers default OFF, and both defaults are measured rather than cautious.
            //
            // PEER (OVERFIT_FP_FLOOR=1): a tie — 206/211, 181/172, 157/156. It could not have been anything
            // else here: the sawtooth amplitude is 6% of a 1.15 GB baseline, 69 MB, and MinAbsoluteGap for
            // memory is 100 MB, so the third gate already filtered everything the floor removes.
            //
            // TREND (OVERFIT_FP_TREND_FLOOR=1): actively WORSE — trend findings 25→45, 27→53, 22→46, and
            // memory trend findings 0→10 where there had been none at all. The reasoning behind the
            // hypothesis was backwards: the sawtooth was not fooling the trend detector, it was protecting
            // it. An oscillating series has rises and falls that cancel, so Theil-Sen's median slope is ~0
            // and tau stays low. The floor removes the oscillation and leaves long flat runs with a few
            // steps in one direction — a highly monotone series, which is precisely what tau rewards.
            //
            // Left as knobs rather than deleted so the measurement stays reproducible.
            var floorGate = Env("OVERFIT_FP_FLOOR", 0) != 0;
            var trendFloorGate = Env("OVERFIT_FP_TREND_FLOOR", 0) != 0;
            var floorLookback = windowSamples;
            var floored = new double[windowSamples];
            var floorScratch = new int[windowSamples + floorLookback];
            var floorsRefused = 0;

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
            var byTrendSignal = new SortedDictionary<string, int>(StringComparer.Ordinal);
            var accused = new HashSet<string>(StringComparer.Ordinal);

            // Per-metric peer detail. A count alone cannot say whether a finding was worth making; the gap and
            // the absolute difference behind it can.
            var peerGaps = new SortedDictionary<MetricIndex, List<double>>();
            var peerDeltas = new SortedDictionary<MetricIndex, List<double>>();
            var peerAbsolute = new SortedDictionary<MetricIndex, List<double>>();

            var buffer = new PeerOutlierFinding[pods];
            var subjects = new IncidentSubject[pods];
            for (var p = 0; p < pods; p++)
            {
                subjects[p] = new IncidentSubject("overfit", "overfit-server", string.Empty, SyntheticCluster.PodName(p), string.Empty);
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

                    var useFloor = floorGate && IsSawtooth(metric);

                    for (var p = 0; p < pods; p++)
                    {
                        var series = cluster.Series(p, metric);
                        double[] values;

                        // The floor needs history BEFORE the window — that is what makes it phase-invariant —
                        // so it reads from the full series rather than from the window slice.
                        if (useFloor && RunningMinimum.TryFloorWindow(
                                series, start, windowSamples, floorLookback, floored, floorScratch))
                        {
                            values = floored.AsSpan().ToArray();
                        }
                        else
                        {
                            floorsRefused += useFloor ? 1 : 0;
                            values = series.AsSpan(start, windowSamples).ToArray();
                        }

                        var work = IsLoadSensitive(metric)
                            ? cluster.Series(p, MetricIndex.RequestsPerSecond).AsSpan(start, windowSamples).ToArray()
                            : [];

                        peers.Add(new PeerSeries(SyntheticCluster.PodName(p), values, work));
                    }

                    var kind = IsLoadSensitive(metric) ? PeerSignalKind.LoadSensitive : PeerSignalKind.LoadIndependent;

                    // The absolute floor is per metric because only the caller knows the units. Set
                    // OVERFIT_FP_ABSOLUTE=0 to measure without it.
                    var peerOptions = PeerOutlierOptions.Balanced with
                    {
                        MinAbsoluteGap = absoluteGate ? AbsoluteFloor(metric) : 0.0
                    };

                    var result = peer.Detect(peers, kind, peerOptions, buffer);

                    var added = pipeline.ObservePeerGroup(
                        metric.ToString(), result, buffer.AsSpan(0, pods), subjects.AsSpan(0, pods), from, to);

                    if (added > 0)
                    {
                        byFamily["peer"] = byFamily.GetValueOrDefault("peer") + added;

                        // The group's own centre, so the relative gap can be turned back into the units an
                        // operator would actually see.
                        var centres = new List<double>(pods);
                        for (var p = 0; p < pods; p++)
                        {
                            var values = peers[p].Values.Span;
                            var finite = new List<double>(values.Length);
                            for (var i = 0; i < values.Length; i++)
                            {
                                if (double.IsFinite(values[i]))
                                {
                                    finite.Add(values[i]);
                                }
                            }

                            if (finite.Count > 0)
                            {
                                finite.Sort();
                                centres.Add(finite[finite.Count / 2]);
                            }
                        }

                        centres.Sort();
                        var groupCentre = centres.Count > 0 ? centres[centres.Count / 2] : double.NaN;

                        for (var p = 0; p < pods; p++)
                        {
                            if (!buffer[p].IsOutlier)
                            {
                                continue;
                            }

                            Record(peerGaps, metric, buffer[p].RelativeGap);
                            Record(peerDeltas, metric, Math.Abs(buffer[p].Comparison.EffectSize));
                            Record(peerAbsolute, metric,
                                double.IsFinite(buffer[p].RelativeGap) && double.IsFinite(groupCentre)
                                    ? buffer[p].RelativeGap * Math.Abs(groupCentre)
                                    : double.NaN);
                        }
                    }

                    // Trend, per pod, over the same window — unless this metric is ablated out.
                    for (var p = 0; p < pods && !trendSkip.Contains(metric); p++)
                    {
                        var history = cluster.Series(p, metric);

                        // The hypothesis under test: on a sawtooth signal, a trend over the raw series is
                        // dominated by where in the tooth the window happens to start and end, not by any
                        // drift. The floor is phase-invariant, so a trend over it should be the leak and
                        // nothing else. OVERFIT_FP_TREND_FLOOR=0 turns it off, which is the other arm.
                        double[] values;

                        if (trendFloorGate && IsSawtooth(metric)
                            && RunningMinimum.TryFloorWindow(
                                history, start, windowSamples, floorLookback, floored, floorScratch))
                        {
                            values = floored.AsSpan().ToArray();
                        }
                        else
                        {
                            values = history.AsSpan(start, windowSamples).ToArray();
                        }

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

                            // Per signal, and only for this family — the whole point of the measurement is
                            // which metric the trend noise sits on, and a total across families hides it.
                            var name = metric.ToString();
                            byTrendSignal[name] = byTrendSignal.GetValueOrDefault(name) + 1;
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
            report.Append($"seasonal     {(seasonal ? "ON" : "off")}\n");
            report.Append($"trend skip   {(trendSkip.Count == 0 ? "(none)" : string.Join(", ", trendSkip))}\n");
            report.Append($"absolute gate {(absoluteGate ? "ON (per-metric floors)" : "off")}\n");
            report.Append($"restarts     {(restarts > 0.0 ? "ON (1/pod/day)" : "ABLATED")}\n");
            report.Append($"sawtooth floor {(floorGate ? $"ON (lookback {floorLookback} samples)" : "off")}"
                          + $"{(floorsRefused > 0 ? $" — {floorsRefused} windows lacked history and fell back" : string.Empty)}\n");
            report.Append($"trend floor  {(trendFloorGate ? $"ON (lookback {floorLookback} samples)" : "off")}\n");
            report.Append($"\nincidents    {incidents}\n");
            report.Append($"findings     {findings}\n");
            report.Append($"pods accused {accused.Count} of {pods}\n");
            // Divided by the EVALUATED stretch, not the generated one. The seasonal warm-up skips the first
            // periods, so dividing by `hours` understated the rate by the warm-up fraction — a measurement tool
            // printing a wrong number is the trap this whole diagnostic exists to avoid.
            var evaluatedHours = Math.Max(
                0.0001, (cluster.Samples - warmup) * cluster.ScrapeSeconds / 3600.0);

            report.Append($"\nevaluated    {evaluatedHours:F1} h of {hours} h generated "
                          + $"({warmup * cluster.ScrapeSeconds / 3600.0:F0} h warm-up skipped)\n");
            report.Append($"rate         {incidents / evaluatedHours:F2} incidents/hour  "
                          + $"({incidents * 24.0 / evaluatedHours:F1} per day)\n");
            report.Append($"             {incidents / (double)evaluations:P1} of evaluations produced one\n");

            if (byFamily.Count > 0)
            {
                report.Append("\nby detector family\n");
                foreach (var (family, count) in byFamily)
                {
                    report.Append($"   {family,-8} {count}\n");
                }
            }

            if (byTrendSignal.Count > 0)
            {
                report.Append("\ntrend findings by signal\n");
                foreach (var (signal, count) in byTrendSignal)
                {
                    report.Append($"   {signal,-24} {count}\n");
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

            if (peerGaps.Count > 0)
            {
                report.Append("\npeer findings in detail — the question a count cannot answer:\n");
                report.Append("was the difference worth reporting, in the metric's own units?\n\n");
                report.Append($"   {"metric",-24}{"count",7}{"med gap",10}{"med delta",11}   median absolute difference\n");

                foreach (var (metric, gaps) in peerGaps)
                {
                    var count = gaps.Count;
                    report.Append($"   {metric,-24}{count,7}{Median(gaps),9:P0}{Median(peerDeltas[metric]),11:F2}"
                                  + $"   {Median(peerAbsolute[metric]):G4}\n");
                }
            }

            _output.WriteLine(report.ToString());

            // Reported, not asserted. The first run of a measurement has no baseline to fail against, and
            // picking a bound now would mean picking one that whatever came out happens to satisfy.
            Assert.True(evaluations > 0, "no window fitted in the generated history");
        }

        /// <summary>
        /// Deferred to <see cref="PeerSignalCatalog"/> rather than restated here.
        ///
        /// <para>This diagnostic used to carry its own list, and that is how memory came to be divided by
        /// request rate in the measurement while the product had no opinion at all: a classification that
        /// lives in a test file is not a fix, it is a second place to be wrong. The catalog is now the single
        /// authority and this measurement exercises it.</para>
        /// </summary>
        private static bool IsLoadSensitive(MetricIndex metric)
        {
            return PeerSignalCatalog.RequiresWork(metric);
        }

        /// <summary>
        /// Signals that climb between garbage collections and drop back at each one, so that an instantaneous
        /// cross-replica comparison compares GC phase rather than health.
        ///
        /// <para>Nothing synchronises collections across replicas, so the phases drift apart and stay apart.
        /// Measured on this very population, the two signals below produced <b>95% of all peer findings</b> on
        /// pods where nothing was wrong, with real median differences of 130 to 480 MB — differences no
        /// threshold can filter, because they are genuine and meaningless at the same time. See
        /// <see cref="RunningMinimum"/>.</para>
        /// </summary>
        private static bool IsSawtooth(MetricIndex metric)
        {
            return metric is MetricIndex.MemoryWorkingSetBytes or MetricIndex.GcGen2HeapBytes;
        }

        /// <summary>
        /// The smallest difference in each signal's own units that anybody would act on.
        ///
        /// <para><b>Derived from what the number means, not from the noise it is measured against.</b> Picking
        /// floors off the healthy spread would be fitting the threshold to the very data the false-positive
        /// rate is then measured on, and the result would be guaranteed rather than earned. Each line below is
        /// an operational claim that can be argued with on its own terms.</para>
        /// </summary>
        private static double AbsoluteFloor(MetricIndex metric)
        {
            return metric switch
            {
                // 1% of wall-clock in GC. Below that, GC is not what is wrong with the pod — and the measured
                // healthy difference was 0.0003, thirty times smaller.
                MetricIndex.GcPauseRatio => 0.01,

                // Nobody pages on a fiftieth of a second. Measured healthy differences: 27 / 77 / 186 ms.
                MetricIndex.LatencyP50Ms => 50.0,
                MetricIndex.LatencyP95Ms => 50.0,
                MetricIndex.LatencyP99Ms => 50.0,

                // Below one request per second apart, two replicas are load-balanced, not anomalous.
                MetricIndex.RequestsPerSecond => 1.0,

                // 100 MB. Smaller differences between replicas are GC phase, not a leak.
                MetricIndex.MemoryWorkingSetBytes => 100e6,
                MetricIndex.GcGen2HeapBytes => 100e6,

                // A ratio of the pod's own limit; a tenth of it is a real difference in headroom.
                MetricIndex.CpuUsageRatio => 0.1,
                MetricIndex.CpuThrottleRatio => 0.05,

                // Counts: one event is the finding. These are also the signals whose group centre is zero, so
                // the relative gate cannot judge them at all and this floor is the only one that applies.
                MetricIndex.ContainerRestarts => 1.0,
                MetricIndex.OomEventsRate => 0.0001,
                MetricIndex.ErrorRate => 0.01,

                MetricIndex.ThreadPoolQueueLength => 5.0,

                _ => 0.0
            };
        }

        private static void Record(SortedDictionary<MetricIndex, List<double>> into, MetricIndex metric, double value)
        {
            if (!into.TryGetValue(metric, out var list))
            {
                list = [];
                into[metric] = list;
            }

            list.Add(value);
        }

        /// <summary>Median of the finite entries; NaN when there are none.</summary>
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

        private static int Env(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : fallback;
        }
    }
}
