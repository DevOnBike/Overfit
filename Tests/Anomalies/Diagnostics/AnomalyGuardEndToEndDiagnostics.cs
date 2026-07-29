// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Rules;
using DevOnBike.Overfit.Anomalies.Rules.Contracts;
using DevOnBike.Overfit.Statistics;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Anomalies.Diagnostics
{
    /// <summary>
    /// The whole guard, end to end, against the live lab: range queries out of Prometheus, through the peer
    /// and trend detectors, into <see cref="IncidentGrouper"/>, out as incidents.
    ///
    /// <para><b>Everything before this was components with unit tests.</b> This is the first thing that asks
    /// the product's actual question — given real metrics from a cluster with one deliberately degraded
    /// replica, does a reader get <i>one incident naming that replica</i>, or a pile of alerts, or nothing at
    /// all. The answer is allowed to be unflattering; the false-positive count on the three healthy replicas
    /// is the number the blueprint's M0 gate turns on and it has never been measured.</para>
    ///
    /// <para>Requires the lab from <c>k8s/</c>, the fault from <c>k8s/overfit/fault-cpu-throttle.yaml</c>, a
    /// port-forward to Prometheus, and — this matters — <b>traffic</b>. The CPU-throttle fault does not
    /// manifest on an idle pod: it was measured at 2.74x on p95 under load and invisible without it.</para>
    /// <code>
    ///   kubectl port-forward -n monitoring svc/overfit-lab-prometheus 9099:9090
    /// </code>
    /// </summary>
    public sealed class AnomalyGuardEndToEndDiagnostics
    {
        /// <summary>Pod-name substring identifying the deliberately degraded replica.</summary>
        private const string DegradedMarker = "degraded";

        private readonly ITestOutputHelper _output;

        public AnomalyGuardEndToEndDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public async Task DetectsTheDegradedReplicaAndGroupsTheFindings()
        {
            var baseUrl = Environment.GetEnvironmentVariable("OVERFIT_LAB_PROMETHEUS")
                          ?? "http://127.0.0.1:9099";
            var windowMinutes = int.TryParse(
                Environment.GetEnvironmentVariable("OVERFIT_LAB_WINDOW_MINUTES"), out var parsed)
                ? parsed
                : 20;

            var end = DateTime.UtcNow;
            var start = end.AddMinutes(-windowMinutes);

            var config = PrometheusHistoricalSourceConfig.ForOverfitServer(
                baseUrl,
                podRegex: "overfit-server-.*",
                namespaceName: "overfit",
                rangeStart: start,
                rangeEnd: end,
                step: TimeSpan.FromSeconds(15));

            using var source = new PrometheusHistoricalSource(config);
            var frames = await source.FetchAsync();

            var history = Collect(frames);

            _output.WriteLine($"window {windowMinutes} min, {frames.Count} scrapes, {history.Count} pods");
            foreach (var (pod, _) in history)
            {
                _output.WriteLine($"   {pod}{(pod.Contains(DegradedMarker, StringComparison.Ordinal) ? "   <-- degraded" : string.Empty)}");
            }

            Assert.True(
                history.Count >= 2,
                $"only {history.Count} pod(s) of history — check the port-forward, the window and that the lab is up.");

            var pipeline = new IncidentPipeline();
            var report = new StringBuilder();

            RunHardRules(history, pipeline, report, start, end);
            RunPeerDetection(history, pipeline, report, start, end);
            RunTrendDetection(history, pipeline, report, start, end);

            _output.WriteLine(report.ToString());

            var incidents = pipeline.Group(IncidentGroupingOptions.Balanced);

            _output.WriteLine($"\n=== {incidents.Count} incident(s) from {pipeline.Count} finding(s) ===");
            for (var i = 0; i < incidents.Count; i++)
            {
                var incident = incidents[i];
                _output.WriteLine($"\n[{i + 1}] {incident.Summary}");
                _output.WriteLine($"    severity {incident.PeakSeverity:F2}, {incident.AffectedSubjects} subject(s), "
                                  + $"{incident.DistinctSignals} signal(s), span {incident.Duration:g}");

                foreach (var finding in incident.Findings)
                {
                    _output.WriteLine($"      - [{finding.Class}] {finding.Signal} @ {finding.Subject.Pod}");
                }
            }

            // Scored, not asserted: the honest question is how much of the noise is on the healthy replicas,
            // and a pass/fail on the first live run would hide that behind a green tick.
            var onDegraded = 0;
            var onHealthy = 0;

            for (var i = 0; i < incidents.Count; i++)
            {
                foreach (var finding in incidents[i].Findings)
                {
                    if (finding.Subject.Pod.Contains(DegradedMarker, StringComparison.Ordinal))
                    {
                        onDegraded++;
                        continue;
                    }

                    onHealthy++;
                }
            }

            _output.WriteLine($"\nfindings on the degraded replica: {onDegraded}");
            _output.WriteLine($"findings on healthy replicas:      {onHealthy}   <-- the false-positive budget");
        }

        /// <summary>
        /// Runs the absolute thresholds, per pod, per metric that has one.
        ///
        /// <para><b>This was missing from the first end-to-end run, and the omission was the point.</b> The
        /// rules family is the only one that reaches CFS throttling — the counters exist solely on containers
        /// carrying a CPU limit, so the peer group held one member and the relative methods were undefined, on
        /// precisely the degraded pod. It is also the only family that catches a single OOM kill: measured, the
        /// peer comparison is blind to one, because a lone event yields a non-zero rate over ~10% of the window
        /// and Cliff's delta then lands under the materiality gate.</para>
        /// </summary>
        private static void RunHardRules(
            SortedDictionary<string, Dictionary<MetricIndex, List<double>>> history,
            IncidentPipeline pipeline,
            StringBuilder report,
            DateTime start,
            DateTime end)
        {
            var rule = new SustainedThresholdRule();
            report.Append("\n=== hard rules (fired only) ===\n");

            var fired = 0;

            foreach (var (pod, byMetric) in history)
            {
                foreach (var (metric, options) in RuleProfiles())
                {
                    if (!byMetric.TryGetValue(metric, out var values) || values.Count == 0)
                    {
                        continue;
                    }

                    var verdict = rule.Evaluate(values.ToArray(), options);

                    if (pipeline.ObserveRule(
                            SubjectFor(pod), NameOf(metric), verdict,
                            new DateTimeOffset(start, TimeSpan.Zero), new DateTimeOffset(end, TimeSpan.Zero),
                            values.ToArray()))
                    {
                        fired++;
                        report.Append($"  {pod,-42} {metric,-24} {verdict.BreachFraction:P0} of window\n");
                    }
                }
            }

            if (fired == 0)
            {
                report.Append("  (none)\n");
            }
        }

        /// <summary>
        /// Which metrics have an absolute threshold worth stating, and which profile. Everything else is left to
        /// the comparative families — an absolute number on latency or memory would be a per-deployment guess.
        /// </summary>
        private static IEnumerable<(MetricIndex Metric, SustainedThresholdOptions Options)> RuleProfiles()
        {
            yield return (MetricIndex.CpuThrottleRatio, SustainedThresholdOptions.ForCpuThrottling);
            yield return (MetricIndex.OomEventsRate, SustainedThresholdOptions.ForRareEvent);
            yield return (MetricIndex.ContainerRestarts, SustainedThresholdOptions.ForRareEvent);
        }

        /// <summary>Runs the peer comparison across all pods, one metric at a time.</summary>
        private static void RunPeerDetection(
            SortedDictionary<string, Dictionary<MetricIndex, List<double>>> history,
            IncidentPipeline pipeline,
            StringBuilder report,
            DateTime start,
            DateTime end)
        {
            var detector = new PeerGroupOutlierDetector();
            report.Append("\n=== peer comparison ===\n");

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;
                var peers = new List<PeerSeries>();
                var subjects = new List<IncidentSubject>();

                foreach (var (pod, byMetric) in history)
                {
                    if (!byMetric.TryGetValue(metric, out var values) || values.Count == 0)
                    {
                        continue;
                    }

                    // Load-sensitive signals are compared per unit of work, which is the entry requirement —
                    // and not a guarantee of comparability. Measured on this very lab, unit cost inverts
                    // under a CPU-throttle fault, so a verdict resting on it alone is not trustworthy.
                    var work = IsLoadSensitive(metric) && byMetric.TryGetValue(MetricIndex.RequestsPerSecond, out var rps)
                        ? rps.ToArray()
                        : Array.Empty<double>();

                    peers.Add(new PeerSeries(pod, values.ToArray(), work));
                    subjects.Add(SubjectFor(pod));
                }

                if (peers.Count < PeerOutlierOptions.Balanced.MinimumPeers)
                {
                    report.Append($"  {metric,-24} skipped: {peers.Count} peer(s)\n");
                    continue;
                }

                var findings = new PeerOutlierFinding[peers.Count];
                var result = detector.Detect(
                    peers,
                    IsLoadSensitive(metric) ? PeerSignalKind.LoadSensitive : PeerSignalKind.LoadIndependent,
                    PeerOutlierOptions.Balanced,
                    findings);

                var added = pipeline.ObservePeerGroup(
                    NameOf(metric), result, findings.AsSpan(0, peers.Count),
                    subjects.ToArray().AsSpan(0, peers.Count),
                    new DateTimeOffset(start, TimeSpan.Zero), new DateTimeOffset(end, TimeSpan.Zero));

                report.Append($"  {metric,-24} {result.Status,-18} high={result.HighCount} low={result.LowCount} -> {added} finding(s)\n");
            }
        }

        /// <summary>Runs the trend detector per pod, per metric.</summary>
        private static void RunTrendDetection(
            SortedDictionary<string, Dictionary<MetricIndex, List<double>>> history,
            IncidentPipeline pipeline,
            StringBuilder report,
            DateTime start,
            DateTime end)
        {
            var detector = new TrendDetector();
            report.Append("\n=== trends (anomalous only) ===\n");

            var anomalous = 0;

            foreach (var (pod, byMetric) in history)
            {
                foreach (var (metric, values) in byMetric)
                {
                    if (values.Count < TrendOptions.Balanced.MinimumSamples)
                    {
                        continue;
                    }

                    var times = new double[values.Count];
                    for (var i = 0; i < times.Length; i++)
                    {
                        times[i] = i * 15.0;
                    }

                    var result = detector.Detect(values.ToArray(), times, TrendOptions.Balanced);

                    if (pipeline.Observe(
                            SubjectFor(pod), NameOf(metric), result,
                            new DateTimeOffset(start, TimeSpan.Zero), new DateTimeOffset(end, TimeSpan.Zero),
                            values.ToArray()))
                    {
                        anomalous++;
                        report.Append($"  {pod,-42} {metric,-24} {result.Direction} tau={result.KendallTau:F2}\n");
                    }
                }
            }

            if (anomalous == 0)
            {
                report.Append("  (none)\n");
            }
        }

        /// <summary>
        /// Reshapes the scrape-ordered frames into per-pod, per-metric series in time order — the shape both
        /// detectors take.
        /// </summary>
        private static SortedDictionary<string, Dictionary<MetricIndex, List<double>>> Collect(
            IReadOnlyList<(long ScrapeTimestampMs, List<RawMetricSeries> Series)> frames)
        {
            var history = new SortedDictionary<string, Dictionary<MetricIndex, List<double>>>(StringComparer.Ordinal);

            for (var f = 0; f < frames.Count; f++)
            {
                foreach (var series in frames[f].Series)
                {
                    var pod = series.Pod.PodName;

                    if (pod.Length == 0 || series.Samples.Count == 0)
                    {
                        continue;
                    }

                    if (!history.TryGetValue(pod, out var byMetric))
                    {
                        byMetric = [];
                        history[pod] = byMetric;
                    }

                    var metric = (MetricIndex)series.MetricTypeId;

                    if (!byMetric.TryGetValue(metric, out var values))
                    {
                        values = [];
                        byMetric[metric] = values;
                    }

                    var value = series.Samples[^1].Value;

                    // Non-finite means the query produced nothing for this pod at this instant. The detectors
                    // filter those themselves, but carrying them keeps the series index-aligned with time.
                    values.Add(value);
                }
            }

            return history;
        }

        /// <summary>
        /// Raw magnitudes that scale with traffic, and are therefore only comparable across peers once
        /// divided by a per-pod work metric. Fractions and per-request measures are already normalised.
        /// </summary>
        private static bool IsLoadSensitive(MetricIndex metric)
        {
            return metric is MetricIndex.CpuUsageRatio
                or MetricIndex.MemoryWorkingSetBytes
                or MetricIndex.GcGen2HeapBytes
                or MetricIndex.GcPauseRatio
                or MetricIndex.ThreadPoolQueueLength;
        }

        /// <summary>
        /// Deployment-owned pod names end in <c>-{replicaSetHash}-{podHash}</c>, so the workload is the name
        /// with those two segments removed. That distinction is what puts the degraded replica in its own
        /// workload while leaving it in the same namespace as its siblings.
        /// </summary>
        private static IncidentSubject SubjectFor(string pod)
        {
            var workload = pod;
            var lastDash = pod.LastIndexOf('-');

            if (lastDash > 0)
            {
                var secondLast = pod.LastIndexOf('-', lastDash - 1);
                if (secondLast > 0)
                {
                    workload = pod[..secondLast];
                }
            }

            return new IncidentSubject("overfit", workload, string.Empty, pod, string.Empty);
        }

        /// <summary>Prometheus-style names, so the incident output reads like the metrics it came from.</summary>
        private static string NameOf(MetricIndex metric)
        {
            return metric switch
            {
                MetricIndex.CpuUsageRatio => "container_cpu_usage_seconds_total",
                MetricIndex.CpuThrottleRatio => "container_cpu_cfs_throttled_periods_total",
                MetricIndex.MemoryWorkingSetBytes => "container_memory_working_set_bytes",
                MetricIndex.OomEventsRate => "container_oom_events_total",
                MetricIndex.LatencyP50Ms => "overfit_chat_response_time_seconds_p50",
                MetricIndex.LatencyP95Ms => "overfit_chat_response_time_seconds_p95",
                MetricIndex.LatencyP99Ms => "overfit_chat_response_time_seconds_p99",
                MetricIndex.RequestsPerSecond => "overfit_chat_requests_total",
                MetricIndex.ErrorRate => "overfit_http_responses_total_5xx_ratio",
                MetricIndex.GcGen2HeapBytes => "dotnet_gc_heap_size_bytes",
                MetricIndex.GcPauseRatio => "dotnet_gc_pause_seconds_total",
                MetricIndex.ThreadPoolQueueLength => "dotnet_threadpool_queue_length",
                MetricIndex.ContainerRestarts => "kube_pod_container_status_restarts_total",
                _ => metric.ToString()
            };
        }
    }
}
