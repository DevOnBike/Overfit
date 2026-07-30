// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Anomalies.Rules;
using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// One evaluation cycle, end to end: a window of metrics in, incidents with identities out, reported to a
    /// sink.
    ///
    /// <para><b>This is the piece that was missing, and its absence was not obvious.</b> Every part below it
    /// existed and was tested — three detector families, a grouper, a tracker, a reporter — and nothing in
    /// <c>Sources/</c> composed them. The only code that ran the whole path was a diagnostic in the test
    /// project, which meant there was no artefact to deploy however finished the parts looked.</para>
    ///
    /// <para><b>No I/O, no timer, no logging.</b> A cycle is a function of a <see cref="MetricWindow"/> and
    /// the options, which is what makes it testable against a recorded window from the real cluster rather
    /// than only against a simulator. Fetching and scheduling belong to the host; the sink is where output
    /// goes.</para>
    ///
    /// <para><b>Coverage is counted, not assumed.</b> A metric this deployment has a query for and which no
    /// pod reported produces no findings — indistinguishable from health at every layer below. The cycle
    /// result carries that count, so a host can say "I am blind" alongside "I see nothing", which are
    /// different statements that silence renders identical.</para>
    ///
    /// <para>Stateful through its <see cref="IncidentTracker"/>. One instance per monitored scope, driven by
    /// one loop; not thread-safe.</para>
    /// </summary>
    public sealed class AnomalyGuard
    {
        private readonly AnomalyGuardOptions _options;
        private readonly IIncidentSink _sink;
        private readonly IIncidentStore? _store;
        private readonly IncidentTracker _tracker;
        private readonly PeerGroupOutlierDetector _peer = new();
        private readonly TrendDetector _trend = new();
        private readonly SustainedThresholdRule _rule = new();

        /// <param name="options">Thresholds, topology and the per-metric floors.</param>
        /// <param name="sink">Where rows go.</param>
        /// <param name="tracking">How incidents are matched across cycles and when they close.</param>
        /// <param name="store">
        /// Optional durable state. Without one, a restart reopens every incident that was running — the
        /// tracker's whole contribution undone by the guard's own rolling update.
        /// </param>
        /// <param name="restoredAt">
        /// Clock used for the staleness bound when restoring. Defaults to now; supplied explicitly by tests,
        /// which must not depend on the wall clock.
        /// </param>
        public AnomalyGuard(
            AnomalyGuardOptions options,
            IIncidentSink sink,
            IncidentTrackingOptions tracking,
            IIncidentStore? store = null,
            DateTimeOffset? restoredAt = null)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(sink);

            _options = options;
            _sink = sink;
            _store = store;
            _tracker = new IncidentTracker(tracking);

            if (store is null)
            {
                return;
            }

            var saved = IncidentStateFormat.Read(store.Load(), out var nextId);

            RestoredIncidents = _tracker.Restore(
                saved, nextId, restoredAt ?? DateTimeOffset.UtcNow, options.MaxRestoredIncidentAge);
        }

        /// <summary>How many incidents were adopted from durable state at construction.</summary>
        public int RestoredIncidents
        {
            get;
        }

        /// <summary>Incidents currently open, including any inside their grace period.</summary>
        public int OpenIncidents => _tracker.OpenCount;

        /// <summary>
        /// Evaluates one window and reports what changed.
        /// </summary>
        /// <param name="window">The evaluated window; <c>NaN</c> where a scrape returned nothing.</param>
        /// <param name="observedAt">Cycle timestamp, used for incident ages.</param>
        public GuardCycleResult RunCycle(MetricWindow window, DateTimeOffset observedAt)
        {
            ArgumentNullException.ThrowIfNull(window);

            var pipeline = new IncidentPipeline();
            var from = window.Start;
            var to = window.End;

            var times = new double[window.Length];
            window.WriteTimestampSeconds(times);

            var blind = 0;
            var partial = 0;

            RunRules(window, pipeline, from, to);

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;
                var reporting = window.PodsReporting(metric);

                if (reporting == 0)
                {
                    blind++;

                    continue;
                }

                if (reporting < window.Pods.Count)
                {
                    partial++;
                }

                RunPeer(window, metric, pipeline, from, to);
                RunTrend(window, metric, times, pipeline, from, to);
            }

            blind += RunCustom(window, times, pipeline, from, to, ref partial);

            var incidents = pipeline.Group(_options.Grouping);
            var tracked = _tracker.Observe(incidents, observedAt);

            var opened = 0;
            var ongoing = 0;
            var resolved = 0;

            for (var i = 0; i < tracked.Count; i++)
            {
                opened += tracked[i].State == IncidentState.Opened ? 1 : 0;
                ongoing += tracked[i].State == IncidentState.Ongoing ? 1 : 0;
                resolved += tracked[i].State == IncidentState.Resolved ? 1 : 0;
            }

            IncidentReporter.Report(tracked, _sink);

            // After reporting, so a crash between the two costs a repeated notification rather than a lost
            // one: an operator told twice is annoyed, an operator never told is unprotected.
            _store?.Save(IncidentStateFormat.Write(_tracker.Snapshot(), _tracker.NextId));

            return new GuardCycleResult(
                pipeline.Count, incidents.Count, opened, ongoing, resolved, blind, partial);
        }

        /// <summary>
        /// The same three families over the deployment's own metrics.
        ///
        /// <para>Everything below the detectors is name-based — <c>SignalFinding.Signal</c> is a string — so a
        /// custom channel reaches the grouper, the tracker and the reporter unchanged. The only thing the
        /// enum was ever needed for is the query catalogue and the learned family's fixed feature set, and a
        /// custom metric touches neither.</para>
        /// </summary>
        private int RunCustom(
            MetricWindow window,
            double[] times,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to,
            ref int partial)
        {
            var blind = 0;

            for (var c = 0; c < _options.CustomMetrics.Count; c++)
            {
                var binding = _options.CustomMetrics[c];
                var reporting = window.PodsReporting(binding.Name);

                if (reporting == 0)
                {
                    blind++;

                    continue;
                }

                if (reporting < window.Pods.Count)
                {
                    partial++;
                }

                if (binding.Rule is { } rule)
                {
                    for (var pod = 0; pod < window.Pods.Count; pod++)
                    {
                        var verdict = _rule.Evaluate(window.Series(pod, binding.Name), rule);

                        pipeline.ObserveRule(
                            Subject(window.Pods[pod]), binding.Name, verdict, from, to, default, binding.Class);
                    }
                }

                RunCustomPeer(window, binding, pipeline, from, to);
                RunCustomTrend(window, binding, times, pipeline, from, to);
            }

            return blind;
        }

        private void RunCustomPeer(
            MetricWindow window,
            CustomMetricBinding binding,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to)
        {
            var podCount = window.Pods.Count;
            var peers = new List<PeerSeries>(podCount);
            var subjects = new IncidentSubject[podCount];

            for (var pod = 0; pod < podCount; pod++)
            {
                var work = binding.SignalKind == PeerSignalKind.LoadSensitive
                    ? window.Series(pod, MetricIndex.RequestsPerSecond).ToArray()
                    : [];

                peers.Add(new PeerSeries(
                    window.Pods[pod], window.Series(pod, binding.Name).ToArray(), work));
                subjects[pod] = Subject(window.Pods[pod]);
            }

            var options = _options.Peer with { MinAbsoluteGap = binding.MinAbsoluteGap };
            var findings = new PeerOutlierFinding[podCount];
            var result = _peer.Detect(peers, binding.SignalKind, options, findings);

            pipeline.ObservePeerGroup(binding.Name, result, findings, subjects, from, to, binding.Class);
        }

        private void RunCustomTrend(
            MetricWindow window,
            CustomMetricBinding binding,
            double[] times,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to)
        {
            var podCount = window.Pods.Count;
            var options = _options.Trend with
            {
                MinAbsoluteChangeOverWindow = binding.MinAbsoluteTrendChange
            };

            var expectation = ReadOnlySpan<double>.Empty;
            double[]? common = null;

            if (_options.DecomposeCommonMode && podCount >= CrossPeerBaseline.MinimumPeers)
            {
                var peers = new List<PeerSeries>(podCount);

                for (var pod = 0; pod < podCount; pod++)
                {
                    peers.Add(new PeerSeries(window.Pods[pod], window.Series(pod, binding.Name).ToArray()));
                }

                common = new double[window.Length];

                if (CrossPeerBaseline.TryBuild(peers, common, new double[podCount]))
                {
                    expectation = common;

                    var verdict = _trend.Detect(common, times, options);

                    pipeline.Observe(
                        WorkloadSubject(), binding.Name, verdict, from, to, common, binding.Class);
                }
            }

            for (var pod = 0; pod < podCount; pod++)
            {
                var series = window.Series(pod, binding.Name).ToArray();
                var verdict = _trend.Detect(series, times, options, double.NaN, expectation);

                pipeline.Observe(
                    Subject(window.Pods[pod]), binding.Name, verdict, from, to, series, binding.Class);
            }
        }

        private void RunRules(
            MetricWindow window,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to)
        {
            for (var r = 0; r < _options.Rules.Count; r++)
            {
                var profile = _options.Rules[r];

                for (var pod = 0; pod < window.Pods.Count; pod++)
                {
                    var verdict = _rule.Evaluate(window.Series(pod, profile.Metric), profile.Options);

                    pipeline.ObserveRule(
                        Subject(window.Pods[pod]), profile.Metric.ToString(), verdict, from, to);
                }
            }
        }

        private void RunPeer(
            MetricWindow window,
            MetricIndex metric,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to)
        {
            var kind = PeerSignalCatalog.Classify(metric);
            var podCount = window.Pods.Count;
            var peers = new List<PeerSeries>(podCount);
            var subjects = new IncidentSubject[podCount];

            for (var pod = 0; pod < podCount; pod++)
            {
                var work = kind == PeerSignalKind.LoadSensitive
                    ? window.Series(pod, MetricIndex.RequestsPerSecond).ToArray()
                    : [];

                peers.Add(new PeerSeries(window.Pods[pod], window.Series(pod, metric).ToArray(), work));
                subjects[pod] = Subject(window.Pods[pod]);
            }

            var options = _options.Peer with
            {
                MinAbsoluteGap = AnomalyGuardOptions.FloorFor(_options.MinAbsoluteGap, metric)
            };

            var findings = new PeerOutlierFinding[podCount];
            var result = _peer.Detect(peers, kind, options, findings);

            pipeline.ObservePeerGroup(metric.ToString(), result, findings, subjects, from, to);
        }

        /// <summary>
        /// Trend, decomposed into what the group did together and what each pod did differently.
        ///
        /// <para>Testing pods in isolation reports a deployment-wide movement once per pod — each finding
        /// correct, none of them about that pod. The common component answers the deployment-level question
        /// once, with no pod named; the residuals answer the per-pod one.</para>
        /// </summary>
        private void RunTrend(
            MetricWindow window,
            MetricIndex metric,
            double[] times,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to)
        {
            var podCount = window.Pods.Count;
            var options = _options.Trend with
            {
                MinAbsoluteChangeOverWindow =
                    AnomalyGuardOptions.FloorFor(_options.MinAbsoluteTrendChange, metric)
            };

            var expectation = ReadOnlySpan<double>.Empty;
            double[]? common = null;

            if (_options.DecomposeCommonMode && podCount >= CrossPeerBaseline.MinimumPeers)
            {
                var peers = new List<PeerSeries>(podCount);

                for (var pod = 0; pod < podCount; pod++)
                {
                    peers.Add(new PeerSeries(window.Pods[pod], window.Series(pod, metric).ToArray()));
                }

                common = new double[window.Length];

                if (CrossPeerBaseline.TryBuild(peers, common, new double[podCount]))
                {
                    expectation = common;

                    var verdict = _trend.Detect(common, times, options);

                    pipeline.Observe(WorkloadSubject(), metric.ToString(), verdict, from, to, common);
                }
            }

            for (var pod = 0; pod < podCount; pod++)
            {
                var series = window.Series(pod, metric).ToArray();
                var verdict = _trend.Detect(series, times, options, double.NaN, expectation);

                pipeline.Observe(Subject(window.Pods[pod]), metric.ToString(), verdict, from, to, series);
            }
        }

        /// <summary>
        /// The subject a pod's findings belong to, with its workload derived from the pod name.
        ///
        /// <para><b>Stamping every pod with one configured workload was a real defect, and it cost an
        /// incident.</b> The grouper scores <c>SameWorkload</c> at 0.7 against a 0.35 threshold, so declaring
        /// four pods to be one workload merges every finding on all of them into a single incident — which is
        /// what happened on the lab, where the deliberately degraded replica is its own Deployment. Told the
        /// truth about topology, the same findings split into two incidents: the healthy three, and the
        /// degraded one alone. <b>The grouper was right; the guard was lying to it.</b></para>
        ///
        /// <para><b>Real topology when it is available, the name heuristic when it is not.</b>
        /// <see cref="AnomalyGuardOptions.PodTopology"/> reads ownership from kube-state-metrics and also
        /// fills the ReplicaSet and node coordinates, which the name cannot. The fallback drops the last two
        /// segments of <c>&lt;deployment&gt;-&lt;replicaset-hash&gt;-&lt;suffix&gt;</c> — right for a
        /// Deployment, wrong for a StatefulSet, a Job or a bare pod.</para>
        /// </summary>
        private IncidentSubject Subject(string pod)
        {
            if (_options.PodTopology is { } topology
                && topology.TryResolve(pod, out var placement)
                && placement.IsKnown)
            {
                return new IncidentSubject(
                    _options.Namespace, placement.Workload, placement.ReplicaSet, pod, placement.Node);
            }

            // Unresolved, so the name heuristic stands in. Note it does NOT return an empty workload: an
            // empty one is shared by every unresolved pod, which would merge all of them into one incident —
            // the exact failure this path exists to avoid.
            return new IncidentSubject(
                _options.Namespace, WorkloadOf(pod), string.Empty, pod, string.Empty);
        }

        /// <summary>
        /// Strips the ReplicaSet hash and pod suffix. Falls back to the configured workload when the name has
        /// too few segments to carry them, which is the case for a bare pod.
        /// </summary>
        private string WorkloadOf(string pod)
        {
            var lastDash = pod.LastIndexOf('-');

            if (lastDash > 0)
            {
                var secondLast = pod.LastIndexOf('-', lastDash - 1);

                if (secondLast > 0)
                {
                    return pod[..secondLast];
                }
            }

            return _options.Workload;
        }

        /// <summary>
        /// The subject of a common-mode finding: the deployment, with <b>no pod</b>. Naming one would be a
        /// false statement — the point of the decomposition is that the movement is not about any replica.
        /// </summary>
        private IncidentSubject WorkloadSubject()
        {
            return new IncidentSubject(
                _options.Namespace, _options.Workload, string.Empty, string.Empty, string.Empty);
        }
    }
}
