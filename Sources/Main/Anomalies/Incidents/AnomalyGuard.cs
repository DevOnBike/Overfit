// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
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
        private readonly LevelShiftDetector _levelShift = new();
        private readonly SustainedThresholdRule _rule = new();

        /// <summary>Signal name silent pods are filed under, stable so a query can group them.</summary>
        private const string SilentPodSignal = "PodReportingNothing";

        private readonly IIncidentStore? _historyStore;

        /// <summary>What this workload normally does, per signal, per hour. Null when history is off.</summary>
        private readonly MetricHistory? _history;

        /// <summary>
        /// What a healthy period looks like in each signal's own units.
        ///
        /// <para><b>It lives here rather than in the host, and that move is the point of this.</b> It used to
        /// sit in the ASP.NET service, which meant the CLI path — the one the lab actually runs — had no
        /// calibration at all, and that the numbers it produced could only ever be copied into a
        /// configuration by hand. A floor that has to be transcribed is a floor that is absent on day one, and
        /// day one with no floors is the configuration measured at 209 false incidents a day.</para>
        /// </summary>
        private readonly FloorCalibrator _calibrator;

        /// <summary>Where the absolute floors come from. Configured first, learned as a fallback.</summary>
        private readonly IAbsoluteFloorSource _floors;

        /// <summary>Which moments were declared abnormal on purpose.</summary>
        private readonly IMaintenanceCalendar _calendar;

        /// <summary>What a healthy period has looked like so far, per signal. Empty until enough is seen.</summary>
        public FloorProposal[] FloorProposals => _calibrator.Propose();

        /// <summary>
        /// The guard's own counters, for a host to expose. <b>Alert on
        /// <c>overfit_guard_last_cycle_timestamp_seconds</c> going stale</b> — it is the one series that makes
        /// "this thing has stopped" visible, and a guard that has stopped is worse than one that never
        /// started, because somebody is relying on it.
        /// </summary>
        public GuardTelemetry Telemetry { get; } = new();

        /// <summary>How many consecutive cycles each known pod has reported nothing.</summary>
        private readonly Dictionary<string, int> _silent = new(StringComparer.Ordinal);

        /// <summary>
        /// Whether the current cycle falls inside a declared maintenance window. Held as a field rather than
        /// threaded through every detector: the answer is a property of the cycle, and passing it down five
        /// call layers to be read in one place would be worse than a field with a short life.
        /// </summary>
        private bool _declaredAbnormal;

        /// <summary>
        /// The workload every subject and every maintenance-window comparison is written against. Starts as
        /// the configured value and is filled in from topology on the first cycle that can resolve it; see
        /// <see cref="ResolveWorkload"/> for why an empty one is not an acceptable resting state.
        /// </summary>
        private string _workload;

        /// <summary>Names the window covering <paramref name="at"/>, or empty when none does.</summary>
        private string SuppressionReason(DateTimeOffset at)
        {
            return _calendar.IsDeclaredAbnormal(at, _workload, out var reason)
                ? reason
                : string.Empty;
        }

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
        /// <param name="historyStore">
        /// Where the per-workload, per-hour baseline survives a restart. <b>Deliberately separate from
        /// <paramref name="store"/></b>: the two payloads have different sizes, different lifetimes and
        /// different formats, and coupling them would mean a version bump in one silently invalidating the
        /// other. Without it the guard still learns within a run and forgets on restart — correct, just
        /// slower to become useful.
        /// </param>
        public AnomalyGuard(
            AnomalyGuardOptions options,
            IIncidentSink sink,
            IncidentTrackingOptions tracking,
            IIncidentStore? store = null,
            DateTimeOffset? restoredAt = null,
            IIncidentStore? historyStore = null)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(sink);

            _options = options;
            _sink = sink;
            _store = store;
            _historyStore = historyStore;
            _tracker = new IncidentTracker(tracking);

            var learned = LearnedState.Read(historyStore?.Load(), options.Trend);
            var history = learned.History;
            var calibrator = learned.Calibrator;

            _history = options.MinimumHistoryDays > 0 ? history : null;
            _calibrator = calibrator;
            Labels = learned.Labels;
            Suppressions = learned.Suppressions;

            // Built here rather than injected, so the default deployment needs nothing but options — and
            // replaceable, because the two things a customer is most likely to own are their own threshold
            // policy and their own deployment calendar.
            _floors = options.Floors ?? new ConfiguredFloorSource(
                options.MinAbsoluteGap, options.MinAbsoluteTrendChange, calibrator,
                options.ApplyCalibratedFloors);

            _calendar = options.Calendar ?? new StaticMaintenanceCalendar(options.MaintenanceWindows);
            _workload = options.Workload;

            // The one contradiction that can be settled before the first cycle: a window scoped to a named
            // workload, no workload configured, and no topology from which one could be derived. Every such
            // window is dead on arrival, and the symptom - being paged during your own declared maintenance -
            // points at the detector rather than at the configuration that caused it.
            if (_workload.Length == 0
                && _calendar.HasWorkloadScopedWindow
                && options.PodTopology is null)
            {
                throw new ArgumentException(
                    "A maintenance window names a workload, but no workload is configured and there is no "
                    + "pod topology to derive one from, so no window can ever match. Set "
                    + nameof(AnomalyGuardOptions.Workload) + ", supply a topology, or scope the window to the "
                    + "namespace by leaving its workload blank.",
                    nameof(options));
            }

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

        /// <summary>
        /// What operators have said about past incidents, and the constraint their <c>--real</c> judgements
        /// place on every future floor proposal.
        ///
        /// <para>Exposed rather than hidden because the host owns the acknowledgement path — the guard
        /// evaluates windows, it does not read a CLI. Adding a label here takes effect on the next proposal
        /// and survives a restart with the rest of the learned state.</para>
        /// </summary>
        public OperatorLabelStore Labels
        {
            get;
        }

        /// <summary>
        /// What an operator has asked not to hear, and until when.
        ///
        /// <para>Every entry expires. A suppression with no end date is a configuration change wearing the
        /// clothes of an acknowledgement — nobody reviews it, nothing reminds anyone it exists, and the pod
        /// most likely to carry one is the pod that eventually breaks.</para>
        /// </summary>
        public SuppressionStore Suppressions
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
        /// <param name="trace">
        /// Optional per-group explanation of why each incident continued or opened. For diagnostics only —
        /// it makes the matcher compute an overlap the ordinary path skips, which is the number that
        /// separates a moved incident centre from a genuinely different group.
        /// </param>
        /// <param name="peerTrace">
        /// Optional per-member explanation of every peer comparison. For diagnostics: "no finding" has five
        /// different causes that call for opposite fixes, and only the individual gates tell them apart.
        /// </param>
        public GuardCycleResult RunCycle(
            MetricWindow window,
            DateTimeOffset observedAt,
            Action<IncidentMatchTrace>? trace = null,
            Action<PeerDecisionTrace>? peerTrace = null)
        {
            ArgumentNullException.ThrowIfNull(window);

            var pipeline = new IncidentPipeline
            {
                Suppressor = Suppressions,
            };

            // Expiry enforced before the cycle rather than trusted: IsSuppressed checks the clock too, but a
            // store that is never pruned grows for as long as the process runs, and the listing an operator
            // reads would fill with entries that mute nothing.
            Suppressions.Prune(observedAt);

            var from = window.Start;
            var to = window.End;

            var times = new double[window.Length];
            window.WriteTimestampSeconds(times);

            var blind = 0;
            var partial = 0;
            var unevaluable = 0;

            // The peer and rule families answer "what is happening"; the trend family answers "where is this
            // going". They need different amounts of time — see AnomalyGuardOptions.RecentWindow — so the
            // caller supplies the long window and the two present-tense families take its tail. The interval
            // travels with them, or a peer finding would report an observation window it never looked at.
            ResolveWorkload(window);

            var recent = RecentSamples(window);
            var recentFrom = window.End - (window.Step * (recent - 1));

            RunRules(window, pipeline, recentFrom, to, recent);
            RunSilentPods(window, pipeline, from, to);

            // Learned from the same window it is about to judge. That is not circular: the floor derived from
            // it applies to LATER cycles, and one window cannot lift a bar that is set from the maximum of
            // hundreds. What it does mean is stated plainly in FloorProposal — a fault inside the observed
            // period raises the bar above itself, so the observed period has to have been healthy.
            _declaredAbnormal = SuppressionReason(observedAt).Length > 0;

            if (!_declaredAbnormal)
            {
                _calibrator.Observe(window);
            }

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

                unevaluable += RunPeer(window, metric, pipeline, recentFrom, to, recent, peerTrace) ? 0 : 1;
                RunTrend(window, metric, times, pipeline, from, to);
            }

            blind += RunCustom(
                window, times, pipeline, from, to, recentFrom, recent, ref partial, ref unevaluable);

            var incidents = pipeline.Group(_options.Grouping);
            var tracked = _tracker.Observe(incidents, observedAt, trace);

            var opened = 0;
            var ongoing = 0;
            var resolved = 0;

            for (var i = 0; i < tracked.Count; i++)
            {
                opened += tracked[i].State == IncidentState.Opened ? 1 : 0;
                ongoing += tracked[i].State == IncidentState.Ongoing ? 1 : 0;
                resolved += tracked[i].State == IncidentState.Resolved ? 1 : 0;
            }

            // Declared abnormal on purpose: reported, flagged, and NOT learned from. See MaintenanceWindow
            // for why the second half matters as much as the first — folding a deployment into "what this
            // cluster does when it is well" takes the one input known to be wrong and treats it as truth.
            var suppressedBy = SuppressionReason(observedAt);

            IncidentReporter.Report(tracked, _sink, suppressedBy);

            // After reporting, so a crash between the two costs a repeated notification rather than a lost
            // one: an operator told twice is annoyed, an operator never told is unprotected.
            _store?.Save(IncidentStateFormat.Write(_tracker.Snapshot(), _tracker.NextId));

            if (_historyStore is not null)
            {
                // Forgotten before saving, so a workload that was deleted stops costing storage on the next
                // restart rather than being carried for ever by a store that only ever grows.
                _history?.Forget(observedAt, TimeSpan.FromDays(MetricHistory.MaxDays * 2));
                _historyStore.Save(LearnedState.Write(
                    _history ?? new MetricHistory(), _calibrator, Labels, Suppressions));
            }

            var result = new GuardCycleResult(
                pipeline.Count, incidents.Count, opened, ongoing, resolved, blind, partial, unevaluable);

            // The guard measuring itself, in the same shape it demands of everything else. Without it, a loop
            // that has stopped or whose queries have started failing produces no incidents — indistinguishable
            // from a healthy cluster, which is the one failure mode this whole subsystem exists to make loud.
            var realLabels = 0;

            for (var i = 0; i < Labels.Labels.Count; i++)
            {
                realLabels += Labels.Labels[i].Kind == OperatorLabelKind.Real ? 1 : 0;
            }

            Telemetry.Feedback(Suppressions.ActiveCount(observedAt), pipeline.Muted, Labels.Count, realLabels);

            Telemetry.Cycle(result, window.Pods.Count, observedAt, _declaredAbnormal);

            return result;
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
            DateTimeOffset recentFrom,
            int recent,
            ref int partial,
            ref int unevaluable)
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
                        var verdict = _rule.Evaluate(
                            Tail(window.Series(pod, binding.Name), recent), rule);

                        pipeline.ObserveRule(
                            Subject(window.Pods[pod]), binding.Name, verdict, recentFrom, to, default,
                            binding.Class);
                    }
                }

                unevaluable += RunCustomPeer(window, binding, pipeline, recentFrom, to, recent) ? 0 : 1;
                RunCustomTrend(window, binding, times, pipeline, from, to);
            }

            return blind;
        }

        /// <summary>
        /// The binding's own floor, falling back to what a healthy period measured for that channel.
        ///
        /// <para>Same precedence as the built-in signals - an explicit value is a decision somebody made and
        /// wins even when it is lower - but for custom channels the fallback did not exist at all, so an
        /// unconfigured binding ran with the gate off. That is the configuration measured at 209 false
        /// incidents a day, reached by default on the metrics the customer added themselves.</para>
        /// </summary>
        private double GapFloor(in CustomMetricBinding binding)
        {
            return binding.MinAbsoluteGap > 0.0
                ? binding.MinAbsoluteGap
                : _floors.MinAbsoluteGap(binding.Name);
        }

        /// <inheritdoc cref="GapFloor"/>
        private double TrendFloor(in CustomMetricBinding binding)
        {
            return binding.MinAbsoluteTrendChange > 0.0
                ? binding.MinAbsoluteTrendChange
                : _floors.MinAbsoluteTrendChange(binding.Name);
        }

        /// <returns>Whether the group reached a verdict; false means nobody was compared at all.</returns>
        private bool RunCustomPeer(
            MetricWindow window,
            CustomMetricBinding binding,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to,
            int recent)
        {
            var podCount = window.Pods.Count;
            var peers = new List<PeerSeries>(podCount);
            var subjects = new IncidentSubject[podCount];

            for (var pod = 0; pod < podCount; pod++)
            {
                var work = binding.SignalKind == PeerSignalKind.LoadSensitive
                    ? TailMemory(window.SeriesMemory(pod, MetricIndex.RequestsPerSecond), recent)
                    : ReadOnlyMemory<double>.Empty;

                peers.Add(new PeerSeries(
                    window.Pods[pod], TailMemory(window.SeriesMemory(pod, binding.Name), recent), work));
                subjects[pod] = Subject(window.Pods[pod]);
            }

            var options = _options.Peer with
            {
                MinAbsoluteGap = GapFloor(binding)
            };
            var findings = new PeerOutlierFinding[podCount];
            var result = _peer.Detect(peers, binding.SignalKind, options, findings);

            pipeline.ObservePeerGroup(binding.Name, result, findings, subjects, from, to, binding.Class);

            return result.Status != DetectionStatus.InsufficientData;
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
            var trendFloor = TrendFloor(binding);
            var options = _options.Trend with
            {
                MinAbsoluteChangeOverWindow = trendFloor
            };

            var expectation = ReadOnlySpan<double>.Empty;
            double[]? common = null;

            if (_options.DecomposeCommonMode && podCount >= CrossPeerBaseline.MinimumPeers)
            {
                var peers = new List<PeerSeries>(podCount);

                for (var pod = 0; pod < podCount; pod++)
                {
                    peers.Add(new PeerSeries(window.Pods[pod], window.SeriesMemory(pod, binding.Name)));
                }

                common = new double[window.Length];

                if (CrossPeerBaseline.TryBuild(peers, common, new double[podCount]))
                {
                    expectation = common;

                    var verdict = _trend.Detect(common, times, options);

                    pipeline.Observe(
                        WorkloadSubject(), binding.Name, verdict, from, to, common, binding.Class);

                    ObserveLevelShift(
                        pipeline, binding.Name, common, from, to, trendFloor, binding.Class);
                }
            }

            for (var pod = 0; pod < podCount; pod++)
            {
                // Custom channels carry their own ceiling on the binding, since the per-metric table is
                // indexed by MetricIndex and cannot hold a name the enum does not have.
                var verdict = _trend.Detect(
                    window.Series(pod, binding.Name),
                    times,
                    options,
                    binding.SaturationLimit,
                    expectation);

                pipeline.Observe(
                    Subject(window.Pods[pod]), binding.Name, verdict, from, to,
                    verdict.Status == DetectionStatus.Anomalous
                        ? window.Series(pod, binding.Name).ToArray()
                        : default,
                    binding.Class);
            }
        }

        private void RunRules(
            MetricWindow window,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to,
            int recent)
        {
            for (var r = 0; r < _options.Rules.Count; r++)
            {
                var profile = _options.Rules[r];

                for (var pod = 0; pod < window.Pods.Count; pod++)
                {
                    var verdict = _rule.Evaluate(
                        Tail(window.Series(pod, profile.Metric), recent), profile.Options);

                    pipeline.ObserveRule(
                        Subject(window.Pods[pod]), profile.Metric.ToString(), verdict, from, to);
                }
            }
        }

        /// <returns>
        /// Whether the group reached a verdict. False means the metric was reported and still produced no
        /// comparison — the silence that reads as health and is not, counted in
        /// <see cref="GuardCycleResult.UnevaluableMetrics"/>.
        /// </returns>
        private bool RunPeer(
            MetricWindow window,
            MetricIndex metric,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to,
            int recent,
            Action<PeerDecisionTrace>? peerTrace = null)
        {
            var kind = PeerSignalCatalog.Classify(metric);
            var podCount = window.Pods.Count;
            var subjects = new IncidentSubject[podCount];
            var groups = new string[podCount];

            for (var pod = 0; pod < podCount; pod++)
            {
                subjects[pod] = Subject(window.Pods[pod]);
                groups[pod] = PeerGroupOf(window.Pods[pod]);
            }

            var options = _options.Peer with
            {
                MinAbsoluteGap = _floors.MinAbsoluteGap(metric)
            };

            // One comparison per declared cohort — see PeerCohorts for why this cannot be inferred and why
            // it costs nothing when nobody has declared anything: every group is then empty, every pod lands
            // together, and this is one Detect call exactly as before.
            var cohorts = PeerCohorts.Partition(groups, podCount);
            var decided = false;

            for (var c = 0; c < cohorts.Count; c++)
            {
                var cohort = cohorts[c];
                var peers = new List<PeerSeries>(cohort.Count);
                var cohortSubjects = new IncidentSubject[cohort.Count];

                for (var i = 0; i < cohort.Count; i++)
                {
                    var pod = cohort[i];

                    var work = kind == PeerSignalKind.LoadSensitive
                        ? TailMemory(window.SeriesMemory(pod, MetricIndex.RequestsPerSecond), recent)
                        : ReadOnlyMemory<double>.Empty;

                    peers.Add(new PeerSeries(
                        window.Pods[pod], TailMemory(window.SeriesMemory(pod, metric), recent), work));
                    cohortSubjects[i] = subjects[pod];
                }

                var findings = new PeerOutlierFinding[cohort.Count];
                var result = _peer.Detect(peers, kind, options, findings);

                decided |= result.Status != DetectionStatus.InsufficientData;

                if (peerTrace is not null)
                {
                    for (var i = 0; i < cohort.Count; i++)
                    {
                        peerTrace(new PeerDecisionTrace(
                            metric.ToString(),
                            result.Status,
                            result.HighCount,
                            result.LowCount,
                            window.Pods[cohort[i]],
                            findings[i].IsOutlier,
                            findings[i].RelativeGap,
                            findings[i].AbsoluteGap,
                            findings[i].Comparison.EffectSize,
                            findings[i].Comparison.PValueCandidateWorse,
                            findings[i].UsableSamples,
                            result.ExcludedCount));
                    }
                }

                pipeline.ObservePeerGroup(metric.ToString(), result, findings, cohortSubjects, from, to);
            }

            return decided;
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
                    _floors.MinAbsoluteTrendChange(metric)
            };

            // Worked out first, because it depends only on the clock and on history — and because it has to
            // reach the WORKLOAD-level trend as well as the per-pod ones. Applying it only to the per-pod
            // trends left the one detector that actually sees a movement shared by every replica judging the
            // raw series, which is the movement a seasonal reference exists to explain.
            var seasonal = Seasonal(metric, window, from);
            var expectation = seasonal;
            double[]? common = null;

            if (_options.DecomposeCommonMode && podCount >= CrossPeerBaseline.MinimumPeers)
            {
                var peers = new List<PeerSeries>(podCount);

                for (var pod = 0; pod < podCount; pod++)
                {
                    peers.Add(new PeerSeries(window.Pods[pod], window.SeriesMemory(pod, metric)));
                }

                common = new double[window.Length];

                if (CrossPeerBaseline.TryBuild(peers, common, new double[podCount]))
                {
                    // The cross-peer component says what the replicas are doing as a group right now, which
                    // removes a difference BETWEEN them and nothing at all from a movement they all share.
                    // Where history can supply the second, it is the better reference and wins.
                    if (expectation.IsEmpty)
                    {
                        expectation = common;
                    }

                    var verdict = _trend.Detect(common, times, options, double.NaN, seasonal);

                    pipeline.Observe(WorkloadSubject(), metric.ToString(), verdict, from, to, common);

                    ObserveLevelShift(
                        pipeline, metric.ToString(), Adjust(common, seasonal), from, to,
                        _floors.MinAbsoluteTrendChange(metric), null);

                }
            }

            // Learned OUTSIDE the decomposition branch, and that placement is the fix rather than a detail.
            // It used to sit inside, so turning DecomposeCommonMode off silently disabled a week of seasonal
            // learning — one option switching off an unrelated subsystem as a side effect nobody would
            // predict from its name. The level comes from the common component when there is one and from
            // the pods' own medians when there is not; the two are close, and a slightly coarser baseline
            // beats no baseline by a distance.
            if (!_declaredAbnormal && _history is not null)
            {
                var level = common is not null ? Median(common) : MedianAcrossPods(window, metric);

                _history.Observe(_workload, metric, from, level);
            }

            for (var pod = 0; pod < podCount; pod++)
            {
                if (IsWarmingUp(window.Pods[pod], to))
                {
                    continue;
                }

                // Judged from the window's own storage. The copy that used to happen here was made for every
                // pod and every signal whether or not anything came of it — two hundred replicas times
                // thirteen channels of eighty samples, several megabytes a cycle, for series that are read
                // once and dropped. The finding is what needs to outlive the window, and on a healthy cluster
                // there are almost none.
                //
                // The ceiling this signal is heading towards, when the operator supplied one. Without it the
                // projection is skipped and the finding reads as it always did; with it, "rose by 11% of
                // typical" becomes "reaches its limit in 40 minutes", which is the difference between an
                // observation and something worth getting up for.
                var verdict = _trend.Detect(
                    window.Series(pod, metric),
                    times,
                    options,
                    AnomalyGuardOptions.LimitFor(_options.SaturationLimit, metric),
                    expectation);

                pipeline.Observe(
                    Subject(window.Pods[pod]), metric.ToString(), verdict, from, to,
                    verdict.Status == DetectionStatus.Anomalous
                        ? window.Series(pod, metric).ToArray()
                        : default);
            }
        }

        /// <summary>
        /// The seasonal expectation for this window, or <paramref name="fallback"/> when there is not enough
        /// history to have one.
        ///
        /// <para><b>Why this beats the cross-peer expectation it replaces.</b> The common component says what
        /// the replicas are doing <i>as a group right now</i>, which removes a difference between replicas and
        /// removes nothing at all from a movement they all share — and the daily traffic curve is exactly such
        /// a movement. Measured on the lab: CPU drift inside a twenty-minute window correlates with traffic
        /// drift at <b>+1.00</b>, and about 10% of windows drift past the trend gate on that alone. A
        /// same-hour-yesterday reference is the only thing that can subtract it.</para>
        ///
        /// <para>Falls back silently and completely. A partial expectation would be worse than none.</para>
        /// </summary>
        private ReadOnlySpan<double> Seasonal(MetricIndex metric, MetricWindow window, DateTimeOffset from)
        {
            if (_history is null || _options.MinimumHistoryDays <= 0)
            {
                return ReadOnlySpan<double>.Empty;
            }

            var expectation = new double[window.Length];

            return _history.TryExpectation(
                _workload, metric, from, window.Step, _options.MinimumHistoryDays, expectation)
                ? expectation
                : ReadOnlySpan<double>.Empty;
        }

        /// <summary>
        /// Subtracts the seasonal expectation, keeping the signal's own scale.
        ///
        /// <para><b>The step detector needs this as much as the trend one does, and finding that out cost two
        /// wrong guesses.</b> Its gate separates a step from a drift by magnitude alone, so a <i>steep enough
        /// ramp</i> reads as a step: a window climbing 60% splits into halves 26% apart at Cliff's delta 1.00,
        /// which clears every gate it has. On a workload that climbs like that every day at noon, that is the
        /// daily curve being reported as a deployment.</para>
        ///
        /// <para><b>The median is added back on purpose.</b> A residual centred on zero has no scale, and the
        /// relative gate downstream would then be dividing by nothing — the same defect documented on
        /// <c>TrendOptions.MinAbsoluteChangeOverWindow</c>, where a series sitting at zero made "cannot judge
        /// the size" read as "the size is large".</para>
        /// </summary>
        private static double[] Adjust(double[] series, ReadOnlySpan<double> expectation)
        {
            if (expectation.IsEmpty || expectation.Length != series.Length)
            {
                return series;
            }

            var level = Median(expectation.ToArray());

            if (!double.IsFinite(level))
            {
                return series;
            }

            var adjusted = new double[series.Length];

            for (var i = 0; i < series.Length; i++)
            {
                adjusted[i] = series[i] - expectation[i] + level;
            }

            return adjusted;
        }

        /// <summary>
        /// The workload's level for one signal when no common component was built — the median across each
        /// pod's own median, which is the same quantity the cross-peer baseline centres on.
        /// </summary>
        private static double MedianAcrossPods(MetricWindow window, MetricIndex metric)
        {
            var pods = window.Pods.Count;
            var medians = new List<double>(pods);

            for (var pod = 0; pod < pods; pod++)
            {
                var median = Median(window.Series(pod, metric).ToArray());

                if (double.IsFinite(median))
                {
                    medians.Add(median);
                }
            }

            if (medians.Count == 0)
            {
                return double.NaN;
            }

            medians.Sort();

            return medians[medians.Count / 2];
        }

        private static double Median(double[] values)
        {
            var finite = new List<double>(values.Length);

            for (var i = 0; i < values.Length; i++)
            {
                if (double.IsFinite(values[i]))
                {
                    finite.Add(values[i]);
                }
            }

            if (finite.Count == 0)
            {
                return double.NaN;
            }

            finite.Sort();

            return finite[finite.Count / 2];
        }

        /// <summary>
        /// Reports pods the cluster says exist and that reported nothing at all.
        ///
        /// <para><b>This is the only check here that looks at what is missing rather than at what was
        /// measured</b>, and it exists because every other family judges a time series. A pod stuck in
        /// <c>Pending</c> or <c>ImagePullBackOff</c>, or crash-looping fast enough to die before its first
        /// scrape, has no series: it is not an outlier, has no trend and breaches no threshold. It is simply
        /// absent from the window — and eleven healthy pods look exactly the same. The guard would report
        /// nothing, which the operator would read as health.</para>
        ///
        /// <para><b>Silence has to persist before it is reported.</b> A pod created just before a cycle
        /// legitimately has no samples yet, and one being deleted stops exporting before the cluster forgets
        /// it. Both clear within a cycle; a rollout that failed does not.</para>
        ///
        /// <para>Skipped entirely when the topology cannot supply a roster, because without one there is no
        /// list of pods that ought to be reporting and the alternative would be inventing it.</para>
        /// </summary>
        private void RunSilentPods(
            MetricWindow window, IncidentPipeline pipeline, DateTimeOffset from, DateTimeOffset to)
        {
            if (_options.SilentPodCycles <= 0 || _options.PodTopology is not IPodRoster roster)
            {
                return;
            }

            // Freshness before contents. The roster keeps its previous snapshot when a refresh fails, so an
            // unreachable Prometheus leaves a list that is confidently wrong in both directions: deleted pods
            // still on it get reported as silent, and pods created since are absent so a failed replica is
            // missed. Declining is the honest outcome — the check has no input, rather than a bad one.
            if (_options.MaxRosterAge > TimeSpan.Zero
                && roster.LastRefreshed is { } refreshed
                && to - refreshed > _options.MaxRosterAge)
            {
                // The counters go with it. They count consecutive cycles of verified silence, and cycles
                // judged against a list nobody could confirm are not that; keeping them would let an outage
                // of the topology query mature into an incident about a pod.
                _silent.Clear();

                return;
            }

            var known = roster.KnownPods;

            if (known.Count == 0)
            {
                // "Nothing known" — not "no pods exist". Treating an empty roster as authoritative would
                // report every pod in the window as unexpected, which is the inverse of this check's job.
                return;
            }

            var reporting = new HashSet<string>(window.Pods, StringComparer.Ordinal);

            for (var i = 0; i < known.Count; i++)
            {
                var pod = known[i];

                if (reporting.Contains(pod))
                {
                    _silent.Remove(pod);

                    continue;
                }

                var cycles = _silent.GetValueOrDefault(pod) + 1;
                _silent[pod] = cycles;

                if (cycles < _options.SilentPodCycles)
                {
                    continue;
                }

                pipeline.ObserveSilentPod(
                    Subject(pod),
                    SilentPodSignal,
                    new SilentPodResult(
                        DetectionStatus.Anomalous,

                        // Counted in cycles rather than minutes: the guard is handed a window, not a
                        // schedule, and only the loop that drives it knows the cadence. Printing an invented
                        // wall-clock figure would be worse than printing none.
                        $"The cluster lists this pod and it has reported no metrics for {cycles} consecutive "
                        + "evaluation cycle(s). Nothing about it was measured, so no other check can see it: "
                        + "a pod that is Pending, cannot pull its image, or restarts before its first scrape "
                        + "is indistinguishable from a pod that does not exist.",
                        cycles,

                        // Climbs with the silence and saturates: ten minutes may still be a slow start, an
                        // hour is a rollout that failed.
                        Math.Clamp(0.5 + (0.05 * cycles), 0.0, 1.0)),
                    from,
                    to);
            }

            // Forget pods the cluster has forgotten, or a scale-down leaves counters growing for ever.
            if (_silent.Count > known.Count)
            {
                var stale = new List<string>();

                foreach (var pod in _silent.Keys)
                {
                    if (!Contains(known, pod))
                    {
                        stale.Add(pod);
                    }
                }

                for (var i = 0; i < stale.Count; i++)
                {
                    _silent.Remove(stale[i]);
                }
            }
        }

        private static bool Contains(IReadOnlyList<string> pods, string pod)
        {
            for (var i = 0; i < pods.Count; i++)
            {
                if (string.Equals(pods[i], pod, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Asks the workload's own aggregate whether it stepped to a new level part-way through the window.
        ///
        /// <para><b>Only on the common component, and only against the workload subject.</b> The case this
        /// covers is every replica moving together, which peer comparison cannot see by construction and the
        /// trend family cannot see either — measured, and the reason is structural rather than a threshold:
        /// Mann-Kendall's tau counts rank order, so a step scores about 0.51 whatever its height, and a 10×
        /// step came back at a <i>worse</i> p-value than a 2.5× one. Running this per pod as well would
        /// duplicate the peer comparison and add its false positives for nothing.</para>
        ///
        /// <para><b>The absolute floor is the TREND gate's, and that was a correction.</b> It originally
        /// borrowed <see cref="AnomalyGuardOptions.MinAbsoluteGap"/> on the argument that both gates ask "how
        /// large a difference in this signal's units matters", one across replicas and one across time, so the
        /// axis should not matter. Four hours on the lab said otherwise: the peer floor for the gen-2 heap
        /// calibrated to <b>0.64 MB</b> and the trend floor to <b>4.23 MB</b> — six times apart, because a GC
        /// sawtooth moves a heap far more over a window than two replicas differ at any instant. At the peer
        /// floor this detector fired about <b>twice an hour on a healthy cluster</b> and became the single
        /// largest remaining source of false positives. Movement over time is the question it asks, so the
        /// gate measured over time is the one it gets.</para>
        /// </summary>
        private void ObserveLevelShift(
            IncidentPipeline pipeline,
            string signal,
            double[] common,
            DateTimeOffset from,
            DateTimeOffset to,
            double minAbsoluteChange,
            SignalClass? signalClass)
        {
            var options = _options.LevelShift with
            {
                MinAbsoluteChange = minAbsoluteChange
            };

            var verdict = _levelShift.Detect(common, options);

            pipeline.ObserveLevelShift(WorkloadSubject(), signal, verdict, from, to, common, signalClass);
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
        /// <summary>
        /// Trailing samples covering <see cref="AnomalyGuardOptions.RecentWindow"/>, never more than the
        /// window holds and never fewer than one. A caller who supplies a window shorter than the recent
        /// window gets the whole of it, which is the pre-existing behaviour.
        /// </summary>
        private int RecentSamples(MetricWindow window)
        {
            if (window.Step <= TimeSpan.Zero || _options.RecentWindow <= TimeSpan.Zero)
            {
                return window.Length;
            }

            var samples = (int)Math.Ceiling(
                _options.RecentWindow.TotalSeconds / window.Step.TotalSeconds);

            return Math.Clamp(samples, 1, window.Length);
        }

        /// <summary>The last <paramref name="samples"/> observations, or all of them if there are fewer.</summary>
        private static ReadOnlySpan<double> Tail(ReadOnlySpan<double> series, int samples)
            => samples >= series.Length ? series : series[^samples..];

        /// <summary>
        /// The last <paramref name="samples"/> of a signal, as memory over the window's own storage.
        ///
        /// <para>The peer detector reads its input inside the call and its findings carry numbers rather than
        /// series, so nothing here outlives the window — which is what makes a slice safe where the trend
        /// path still has to copy. It was the last per-pod allocation in a cycle: two copies of the tail per
        /// replica per signal, made whether or not anything came of them.</para>
        /// </summary>
        private static ReadOnlyMemory<double> TailMemory(ReadOnlyMemory<double> series, int samples)
            => samples >= series.Length ? series : series[^samples..];

        /// <summary>
        /// The cohort this pod may be compared within, from the topology. Empty when nothing was declared,
        /// which puts every pod in one group — the behaviour before cohorts existed.
        /// </summary>
        private string PeerGroupOf(string pod)
        {
            if (_options.PodTopology is { } topology
                && topology.TryResolve(pod, out var placement)
                && placement.PeerGroup is { Length: > 0 } group)
            {
                return group;
            }

            return string.Empty;
        }

        /// <summary>
        /// Whether <paramref name="pod"/> is too young for the trend family to have an opinion about it.
        ///
        /// <para>See <see cref="AnomalyGuardOptions.WarmUpGrace"/> for the measurement behind this. Three
        /// things make it fail closed rather than open: no configured grace means no exemption, no topology
        /// means no exemption, and an <b>unknown</b> creation time means no exemption. A pod is only spared
        /// when the cluster positively states that it is new.</para>
        /// </summary>
        private bool IsWarmingUp(string pod, DateTimeOffset at)
        {
            if (_options.WarmUpGrace <= TimeSpan.Zero
                || _options.PodTopology is not { } topology
                || !topology.TryResolve(pod, out var placement)
                || placement.CreatedAt == default)
            {
                return false;
            }

            return at - placement.CreatedAt < _options.WarmUpGrace;
        }

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

            return _workload;
        }

        /// <summary>
        /// The subject of a common-mode finding: the deployment, with <b>no pod</b>. Naming one would be a
        /// false statement — the point of the decomposition is that the movement is not about any replica.
        /// </summary>
        private IncidentSubject WorkloadSubject()
        {
            return new IncidentSubject(
                _options.Namespace, _workload, string.Empty, string.Empty, string.Empty);
        }

        /// <summary>
        /// Fills in the workload from the cluster's own answer when configuration did not state one.
        ///
        /// <para><b>Deriving beats defaulting to empty, and the difference is two silent failures.</b> An empty
        /// workload makes every workload-scoped maintenance window unmatchable, and it collapses the incident
        /// tracker's subject key to <c>"namespace/"</c> - so a memory incident that closed and a CPU incident
        /// that opened are reported as one continuing problem. kube-state-metrics already knows the owner of
        /// every pod and <see cref="IPodTopology"/> already reads it, so the answer costs no new query.</para>
        ///
        /// <para>The most common owner across the window, not the first: a namespace can hold more than one
        /// deployment, and the majority is the one this guard's scope is about. Resolved once and kept - it
        /// keys the seasonal history, and a value that moved between cycles would split a workload's learned
        /// baseline across two names.</para>
        /// </summary>
        private void ResolveWorkload(MetricWindow window)
        {
            if (_workload.Length > 0 || _options.PodTopology is not { } topology)
            {
                return;
            }

            var counts = new Dictionary<string, int>(StringComparer.Ordinal);

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                if (!topology.TryResolve(window.Pods[pod], out var placement)
                    || !placement.IsKnown
                    || placement.Workload.Length == 0)
                {
                    continue;
                }

                counts[placement.Workload] = counts.GetValueOrDefault(placement.Workload) + 1;
            }

            var best = string.Empty;
            var bestCount = 0;

            foreach (var (name, count) in counts)
            {
                // Ties broken by name, so the resolved workload does not depend on dictionary ordering.
                if (count > bestCount || (count == bestCount && string.CompareOrdinal(name, best) < 0))
                {
                    best = name;
                    bestCount = count;
                }
            }

            _workload = best;
        }
    }
}
