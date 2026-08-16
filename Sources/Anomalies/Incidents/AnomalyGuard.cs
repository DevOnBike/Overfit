// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Rules;
using DevOnBike.Overfit.Runtime;
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
        private readonly IClock _clock;
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

        /// <summary>How many reported incidents stay acknowledgeable. See <see cref="_recent"/>.</summary>
        private const int MaxRecentIncidents = 400;

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

        /// <summary>
        /// How far each replica has sat from its peers over recent cycles, so a difference it has held since
        /// it started stops being reported as news. Null when
        /// <see cref="AnomalyGuardOptions.PeerNovelty"/> was not configured, which leaves the peer family
        /// exactly as it was.
        /// </summary>
        private readonly PeerNoveltyTracker? _novelty;

        /// <summary>Which moments were declared abnormal on purpose.</summary>
        private readonly IMaintenanceCalendar _calendar;

        /// <summary>
        /// What a healthy period has looked like so far, per signal. Empty until enough is seen.
        ///
        /// <para><b>Takes <see cref="_gate"/>, which it did not.</b> <c>FloorCalibrator</c> says plainly
        /// that it is not thread-safe, and <c>Propose</c> is not a pure read: it computes and caches, and
        /// a cycle running concurrently invalidates that cache and refills the accumulators underneath it.
        /// Nothing calls this yet, which is exactly why it was easy to miss — the lock exists because the
        /// acknowledgement endpoint made this class cross-thread for the first time, and a property that
        /// skips it is a trap laid for whoever wires the next reader.</para>
        /// </summary>
        public FloorProposal[] FloorProposals
        {
            get
            {
                lock (_gate)
                {
                    return _calibrator.Propose();
                }
            }
        }

        /// <summary>
        /// The guard's own counters, for a host to expose. <b>Alert on
        /// <c>overfit_guard_last_cycle_timestamp_seconds</c> with <c>absent()</c> as well as a staleness
        /// comparison</b> — the comparison alone goes <c>inactive</c> when the pod disappears, because the
        /// series disappears with it. Measured 2026-08-05; see <c>k8s/lab/guard-alerts.yaml</c> for the rule
        /// and the evidence. A guard that has stopped is worse than one that never started, because somebody
        /// is relying on it.
        /// </summary>
        public GuardTelemetry Telemetry
        {
            get;
        }

        /// <summary>How many consecutive cycles each known pod has reported nothing.</summary>
        private readonly Dictionary<string, int> _silent = new(StringComparer.Ordinal);

        /// <summary>
        /// How many consecutive cycles each pod has been a peer outlier, per persistence-gated custom channel.
        ///
        /// <para><b>Separate from <see cref="_silent"/> on purpose.</b> The two track different predicates —
        /// reporting nothing at all, and standing apart from peers on a continuous value — and a pod can be in
        /// either independently. Sharing one counter would mean a pod sliding from degraded into total silence
        /// resets evidence that should carry forward, or the reverse.</para>
        ///
        /// <para>Not persisted across restarts, matching <see cref="_silent"/>: a restart re-earns its
        /// cycles.</para>
        /// </summary>
        private readonly Dictionary<string, Dictionary<string, int>> _customBreach =
            new(StringComparer.Ordinal);

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

        /// <summary>
        /// Serialises a cycle against an acknowledgement arriving from the host's HTTP thread.
        ///
        /// <para>The stores in this subsystem are single-threaded by design and say so, and an
        /// acknowledgement is the first thing that ever wanted to touch them from somewhere else. A lock
        /// around a cycle is cheap — a cycle is arithmetic over a window already in memory, milliseconds, no
        /// I/O — and it keeps every store's contract intact instead of making four of them thread-safe for
        /// one rare caller.</para>
        /// </summary>
        private readonly object _gate = new();

        /// <summary>
        /// What each recently reported incident was about, so an acknowledgement carrying only an identifier
        /// can be turned into a label with a magnitude.
        ///
        /// <para><b>In memory rather than in the durable state, deliberately.</b> An operator acknowledges an
        /// incident they can see, which is one this process reported; carrying the index across restarts
        /// would mean widening the persisted incident format for a lookup that is only useful while somebody
        /// is looking. An identifier this guard has never reported is refused with a message that says so,
        /// which is a better answer than a label about a magnitude nobody can vouch for.</para>
        /// </summary>
        private readonly Dictionary<long, SignalFinding> _recent = [];

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
        /// <param name="clock">
        /// Time source. Defaults to <see cref="SystemClock"/>.
        ///
        /// <para><b>Read in exactly one place</b>: it supplies the staleness bound when restoring from
        /// <paramref name="store"/>, and only when <paramref name="restoredAt"/> was not given
        /// (<c>restoredAt ?? clock.UtcNow</c>). Nothing else in the guard reads it — incident ages come from
        /// the <c>observedAt</c> handed to <see cref="RunCycle"/>, which is a parameter rather than a clock
        /// read precisely so a replay can drive the same history twice and get the same answer. Injecting a
        /// clock here therefore changes restore behaviour and nothing else.</para>
        /// </param>
        public AnomalyGuard(
            AnomalyGuardOptions options,
            IIncidentSink sink,
            IncidentTrackingOptions tracking,
            IIncidentStore? store = null,
            DateTimeOffset? restoredAt = null,
            IIncidentStore? historyStore = null,
            IClock? clock = null)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(sink);

            _clock = clock ?? SystemClock.Instance;
            _options = options;
            _sink = sink;
            _store = store;
            _historyStore = historyStore;
            _tracker = new IncidentTracker(tracking);

            // Labelled at construction, because a scope's identity cannot change while it runs and a
            // telemetry instrument whose label moves is worse than one with none — every alert written
            // against the old value goes quiet without erroring.
            Telemetry = new GuardTelemetry(options.Scope);

            var learned = LearnedState.Read(historyStore?.Load(), options.Trend);
            var history = learned.History;
            var calibrator = learned.Calibrator;

            _history = options.MinimumHistoryDays > 0 ? history : null;
            _calibrator = calibrator;

            // After the learned-state load and before anything reads a proposal, because a restored
            // calibrator carries the accumulators of a previous run and would otherwise answer one question
            // from them before being told which channels it must not answer for at all.
            _calibrator.ExemptFromCalibration(options.NonCalibratedCustomChannels);

            Labels = learned.Labels;
            Suppressions = learned.Suppressions;

            // Built here rather than injected, so the default deployment needs nothing but options — and
            // replaceable, because the two things a customer is most likely to own are their own threshold
            // policy and their own deployment calendar.
            _floors = options.Floors ?? new ConfiguredFloorSource(
                options.MinAbsoluteGap, options.MinAbsoluteTrendChange, options.MinAbsoluteStepChange,
                calibrator, options.ApplyCalibratedFloors);

            _calendar = options.Calendar ?? new StaticMaintenanceCalendar(options.MaintenanceWindows);
            _workload = options.Workload;
            _novelty = RestoreNovelty(options, learned.PeerNovelty);

            // The one contradiction that can be settled before the first cycle: a window scoped to a named
            // workload, no workload configured, and no topology from which one could be derived. Every such
            // window is dead on arrival, and the symptom - being paged during your own declared maintenance -
            // points at the detector rather than at the configuration that caused it.
            if (_workload.Length == 0
                && _calendar.HasWorkloadScopedWindow
                && options.PodTopology == null)
            {
                throw new ArgumentException(
                    "A maintenance window names a workload, but no workload is configured and there is no "
                    + "pod topology to derive one from, so no window can ever match. Set "
                    + nameof(AnomalyGuardOptions.Workload) + ", supply a topology, or scope the window to the "
                    + "namespace by leaving its workload blank.",
                    nameof(options));
            }

            if (store != null)
            {
                var saved = IncidentStateFormat.Read(store.Load(), out var nextId);

                RestoredIncidents = _tracker.Restore(
                    saved, nextId, restoredAt ?? _clock.UtcNow, options.MaxRestoredIncidentAge);
            }

            // After both loads, and reached even when there is no incident store, because the learned-state
            // load happened either way. A start that could not read its state is the moment this matters
            // most: what follows is a cold start, which is indistinguishable from a healthy first run at
            // every layer above — the guard reports nothing unusual while running with none of its history.
            RefreshStateError();
        }

        /// <summary>How many incidents were adopted from durable state at construction.</summary>
        public int RestoredIncidents
        {
            get;
        }

        /// <summary>
        /// Builds the novelty tracker, or refuses to start when it was asked for without the floor it needs.
        ///
        /// <para><b>Refusing is the point.</b> This is a gate that makes the guard <i>quieter</i>, and its
        /// change floor has never been measured — see
        /// <see cref="AnomalyGuardOptions.MinAbsoluteGapChange"/>. A missing floor would leave the gate
        /// running on the relative test alone, which is a threshold nobody chose applied to a suppression
        /// decision. Starting loudly beats suppressing quietly.</para>
        ///
        /// <para><b>Restored with no roster, deliberately.</b> The ADR's reused-pod-name check is enforced on
        /// the first observation instead — <c>PeerNoveltyTracker</c> resets a pod's history the moment the
        /// cluster reports a creation time that differs from the recorded one. That covers the case a
        /// load-time check cannot: topology is frequently unavailable at construction, and refusing every row
        /// then would make persistence worthless.</para>
        /// </summary>
        private static PeerNoveltyTracker? RestoreNovelty(AnomalyGuardOptions options, string state)
        {
            if (options.PeerNovelty is not { } novelty)
            {
                return null;
            }

            if (options.MinAbsoluteGapChange is not { } floors || floors.Count < (int)MetricIndex.Count)
            {
                throw new ArgumentException(
                    "The peer-novelty gate is configured but "
                    + nameof(AnomalyGuardOptions.MinAbsoluteGapChange)
                    + " is missing or shorter than MetricIndex.Count. The change-in-gap floor has no measured "
                    + "default and none is invented: a suppression gate running with its floor off would "
                    + "silence findings against a threshold nobody chose.",
                    nameof(options));
            }

            for (var c = 0; c < options.CustomMetrics.Count; c++)
            {
                var binding = options.CustomMetrics[c];

                if (binding.MinAbsoluteGapChange > 0.0)
                {
                    continue;
                }

                throw new ArgumentException(
                    $"The peer-novelty gate is configured and custom channel '{binding.Name}' carries no "
                    + nameof(CustomMetricBinding.MinAbsoluteGapChange)
                    + ". Every peer-evaluated channel needs one, and a custom channel has no per-metric table "
                    + "to fall back on — zero there is indistinguishable from 'not supplied'.",
                    nameof(options));
            }

            return PeerNoveltyTracker.Read(state, novelty);
        }

        /// <summary>
        /// Why durable state could not be read or written, or <c>null</c> when the last attempt succeeded.
        ///
        /// <para><b>The failure this exposes is invisible by every other route.</b> The stores swallow their
        /// exceptions on purpose — detection that stops because a volume filled up has replaced the problem
        /// it was bought to detect — so cycles keep completing, incidents keep being reported, and nothing
        /// looks wrong. The bill arrives at the next restart, when every open incident reopens at once and
        /// a week of calibration is gone. An operator watching <c>overfit_guard_state_failures_total</c>
        /// learns hours earlier.</para>
        ///
        /// <para>Set at construction from the loads, and re-evaluated after the saves in every cycle. The
        /// incident store is named first when both have failed, because losing incident identity is felt
        /// immediately and losing learned state is felt slowly.</para>
        /// </summary>
        public string? StateError
        {
            get; private set;
        }

        /// <summary>
        /// Refreshes <see cref="StateError"/> from both stores and counts a failure when there is one.
        ///
        /// <para>One increment per check, not one per store: the counter answers "how many cycles could not
        /// persist", and a full volume that fails both writes is one such cycle, not two. A host alerting on
        /// the rate would otherwise see a step change from a configuration that added a second store.</para>
        /// </summary>
        private void RefreshStateError()
        {
            StateError = _store?.LastError ?? _historyStore?.LastError;

            if (StateError != null)
            {
                Telemetry.StateWriteFailed();
            }
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
        /// <param name="trendTrace">
        /// Optional per-pod, per-signal explanation of every trend decision, built-in channels and custom ones
        /// alike. For diagnostics.
        ///
        /// <para><b>Invoked for pods the trend family skipped as well as those it judged</b> — a pod inside
        /// the warm-up grace is reported with <c>WarmingUp</c> set and its measured fields left at zero, which
        /// is the case this callback was worth adding for. Without it, "tested and found healthy" and "never
        /// tested" were the same silence, and during a rollout the second is every pod.</para>
        /// </param>
        /// <param name="ruleTrace">
        /// Optional per-pod explanation of every absolute-threshold decision, over
        /// <c>AnomalyGuardOptions.Rules</c> and over every <c>CustomMetricBinding</c> carrying a <c>Rule</c>
        /// alike. For diagnostics.
        ///
        /// <para><b>What an empty trace means, which is the whole point of the callback.</b> A row is emitted
        /// for every configured rule against every pod, for every verdict — <c>Anomalous</c>,
        /// <c>Healthy</c>, <c>WarmingUp</c> and the <c>InsufficientData</c> of a channel nobody reported. So a
        /// trace that came back empty says <b>no rule is configured on any channel</b> (or the window carried
        /// no pods), and nothing else. It is not evidence that a rule ran and found nothing — that case has
        /// its own rows.</para>
        ///
        /// <para><b>CORRECTION, recorded rather than quietly rewritten.</b> Until <c>AN-D14</c> this covered
        /// only the built-in metrics: a custom binding's rule was evaluated by the same detector into the same
        /// pipeline and emitted no row, so an empty trace meant "no BUILT-IN rule was configured" while
        /// reading as "no rule fired" — and on a deployment whose rules are all custom, which the lab's
        /// channels are, it was empty on every cycle. That is the silence-reads-as-health shape this
        /// subsystem is organised against, arriving through the diagnostic instead of the detector.</para>
        ///
        /// <para>A channel with no rule configured is not evaluated by this family at all and produces no row.
        /// That is the one remaining absence, and it is a statement about the configuration rather than about
        /// the cluster.</para>
        ///
        /// <para><b>Empty is not the same as absent.</b> A caller passing <c>null</c> asked for no trace and
        /// gets no rows because none were built; a caller passing a sink and receiving nothing has been told
        /// something about the configuration. Nothing downstream can tell those apart from the rows alone, so
        /// a diagnostic that reports this trace should say whether it was collected.</para>
        /// </param>
        public GuardCycleResult RunCycle(
            MetricWindow window,
            DateTimeOffset observedAt,
            Action<IncidentMatchTrace>? trace = null,
            Action<PeerDecisionTrace>? peerTrace = null,
            Action<TrendDecisionTrace>? trendTrace = null,
            Action<RuleDecisionTrace>? ruleTrace = null)
        {
            ArgumentNullException.ThrowIfNull(window);

            lock (_gate)
            {
                return RunCycleCore(window, observedAt, trace, peerTrace, trendTrace, ruleTrace);
            }
        }

        /// <summary>
        /// Acknowledges an incident an operator has looked at.
        ///
        /// <para><b>Both effects, or the feature is one of the two halves that do not work alone.</b> The
        /// suppression is the relief the operator needs in seconds; the label is what makes the threshold
        /// right in days. A <c>--real</c> acknowledgement records only the label, and that label then caps
        /// every future floor proposal for the signal — the constraint that stops a hundred honest
        /// dismissals from converging on a guard that reports nothing.</para>
        ///
        /// <para>The configured floor is never touched. That stays a human decision in a file somebody
        /// reviews; a production threshold moved by a sample of one is how a guard goes blind to a real
        /// fault at exactly the size somebody once dismissed.</para>
        /// </summary>
        /// <param name="incidentId">An incident this guard has reported since it started.</param>
        /// <param name="kind">Whether the operator judged it noise or real.</param>
        /// <param name="mute">
        /// How long to stop reporting this signal on this subject. Ignored for
        /// <see cref="OperatorLabelKind.Real"/> — silencing something an operator just confirmed is a
        /// contradiction, and accepting it quietly would be worse than refusing it.
        /// </param>
        /// <param name="reason">What the operator typed.</param>
        /// <param name="now">Clock, supplied so tests do not depend on the wall clock.</param>
        /// <returns>What was recorded, for the caller to echo back.</returns>
        /// <exception cref="ArgumentException">The identifier is not one this guard has reported.</exception>
        public string Acknowledge(
            long incidentId,
            OperatorLabelKind kind,
            TimeSpan? mute,
            string reason,
            DateTimeOffset now)
        {
            ArgumentNullException.ThrowIfNull(reason);

            lock (_gate)
            {
                if (!_recent.TryGetValue(incidentId, out var finding))
                {
                    throw new ArgumentException(
                        $"Incident {incidentId} is not one this guard has reported since it started. Only "
                        + "incidents it has seen can be acknowledged, because the label has to carry the "
                        + "finding's size in the signal's own units and nothing else knows it.",
                        nameof(incidentId));
                }

                Labels.Add(new OperatorLabel(
                    incidentId, finding.Signal, kind, finding.Magnitude, now, reason));

                _calibrator.UseLabels(Labels);

                if (kind == OperatorLabelKind.Real || mute is not { } window)
                {
                    return string.Create(
                        CultureInfo.InvariantCulture,
                        $"recorded {kind} on {finding.Signal} "
                        + $"(magnitude {finding.Magnitude:G4}); no suppression opened");
                }

                Suppressions.Add(
                    new SignalSuppression(
                        finding.Subject.Pod,
                        finding.Subject.Workload,
                        finding.Signal,
                        now + window,
                        incidentId,
                        reason,
                        finding.Magnitude),
                    now);

                var who = finding.Subject.Pod.Length > 0 ? finding.Subject.Pod : finding.Subject.Workload;

                return string.Create(
                    CultureInfo.InvariantCulture,
                    $"recorded {kind} on {finding.Signal} (magnitude {finding.Magnitude:G4}) and muted it "
                    + $"for {who} until {now + window:u}");
            }
        }

        /// <summary>The suppressions an operator can read back, copied out under the cycle lock.</summary>
        public SignalSuppression[] ActiveSuppressions(DateTimeOffset at)
        {
            lock (_gate)
            {
                var active = new List<SignalSuppression>(Suppressions.Count);

                for (var i = 0; i < Suppressions.Suppressions.Count; i++)
                {
                    if (Suppressions.Suppressions[i].IsActive(at))
                    {
                        active.Add(Suppressions.Suppressions[i]);
                    }
                }

                return active.ToArray();
            }
        }

        private GuardCycleResult RunCycleCore(
            MetricWindow window,
            DateTimeOffset observedAt,
            Action<IncidentMatchTrace>? trace,
            Action<PeerDecisionTrace>? peerTrace,
            Action<TrendDecisionTrace>? trendTrace = null,
            Action<RuleDecisionTrace>? ruleTrace = null)
        {

            var pipeline = new IncidentPipeline
            {
                Suppressor = Suppressions,
                StandingSeverityScale = _options.PeerNovelty?.StandingSeverityScale ?? 1.0,
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

            RunRules(window, pipeline, recentFrom, to, recent, ruleTrace);
            RunSilentPods(window, pipeline, from, to);
            PruneNovelty();

            _declaredAbnormal = SuppressionReason(observedAt).Length > 0;

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
                RunTrend(window, metric, times, pipeline, from, to, trendTrace);
            }

            blind += RunCustom(
                window, times, pipeline, from, to, recentFrom, recent, ref partial, ref unevaluable,
                peerTrace, trendTrace, ruleTrace);

            // AFTER every detector, and the position is the whole point rather than a tidying-up.
            //
            // The comment that used to sit above this call — at the TOP of the cycle — said "the floor
            // derived from it applies to LATER cycles". That was the intent and it was not what the code
            // did: `Observe` invalidates the proposal cache, so every gate below then read a floor that
            // already contained the window it was about to judge. With the proposal set from the MAXIMUM
            // times a 1.25 margin, the floor was therefore never below 1.25x the very quantity being gated,
            // and the gate could not fire — not "rarely", but never, for any fault, once thirty samples
            // existed.
            //
            // It went unnoticed for as long as the floor was calibrated on a DIFFERENT quantity from the one
            // it gated: a per-pod slope against a cross-pod step is a loose enough relationship that the
            // inequality did not always hold. Fixing that mismatch is what made the self-reference exact and
            // therefore visible — a repair that exposed the defect it was standing next to.
            //
            // Moved here, a cycle is judged only against what earlier cycles measured. A sustained fault
            // still raises the bar for the cycles that follow, which is the hazard FloorProposal documents
            // and the reason calibration is meant to run over a period somebody has confirmed was healthy.
            if (!_declaredAbnormal)
            {
                _calibrator.Observe(window);
            }

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

            // Indexed before reporting, so an operator acting on a row they have just seen finds it here.
            // Bounded, and the oldest go first: the index exists to serve somebody looking at a screen, and
            // nobody acknowledges an incident from four hundred incidents ago.
            for (var i = 0; i < tracked.Count; i++)
            {
                _recent[tracked[i].Id] = tracked[i].Incident.Primary;
            }

            while (_recent.Count > MaxRecentIncidents)
            {
                var oldest = long.MaxValue;

                foreach (var id in _recent.Keys)
                {
                    oldest = Math.Min(oldest, id);
                }

                _recent.Remove(oldest);
            }

            IncidentReporter.Report(tracked, _sink, suppressedBy);

            // After reporting, so a crash between the two costs a repeated notification rather than a lost
            // one: an operator told twice is annoyed, an operator never told is unprotected.
            _store?.Save(IncidentStateFormat.Write(_tracker.Snapshot(), _tracker.NextId));

            if (_historyStore != null)
            {
                // Forgotten before saving, so a workload that was deleted stops costing storage on the next
                // restart rather than being carried for ever by a store that only ever grows.
                _history?.Forget(observedAt, TimeSpan.FromDays(MetricHistory.MaxDays * 2));
                _historyStore.Save(LearnedState.Write(
                    _history ?? new MetricHistory(), _calibrator, Labels, Suppressions, _novelty));
            }

            RefreshStateError();

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
            ref int unevaluable,
            Action<PeerDecisionTrace>? peerTrace = null,
            Action<TrendDecisionTrace>? trendTrace = null,
            Action<RuleDecisionTrace>? ruleTrace = null)
        {
            var blind = 0;

            for (var c = 0; c < _options.CustomMetrics.Count; c++)
            {
                var binding = _options.CustomMetrics[c];
                var reporting = window.PodsReporting(binding.Name);

                if (reporting == 0)
                {
                    blind++;

                    // Traced BEFORE the continue, and this branch is the half the obvious fix misses. A
                    // binding whose channel nobody reported skips the evaluation below entirely, so threading
                    // the callback into the rule loop alone would still leave a configured rule producing no
                    // row at all — and an empty trace would keep meaning two things on exactly the cycles
                    // where the operator is looking at it.
                    TraceUnreportedRule(window, binding, ruleTrace);

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

                        // The same row the built-in family emits in RunRules, for every verdict rather than
                        // only the findings. AN-D14: this path evaluated the same detector into the same
                        // pipeline and emitted nothing, so on a deployment whose rules are ALL custom — which
                        // the lab's channels are — the trace was empty every cycle and read as "no rule
                        // fired".
                        ruleTrace?.Invoke(new RuleDecisionTrace(
                            binding.Name,
                            window.Pods[pod],
                            verdict.Status,
                            rule.Threshold,
                            rule.MinBreachFraction,
                            verdict.BreachFraction,
                            verdict.BreachedSamples,
                            verdict.UsableSamples,
                            verdict.PeakValue,
                            verdict.MedianValue,
                            verdict.Reason));

                        pipeline.ObserveRule(
                            Subject(window.Pods[pod]), binding.Name, verdict, recentFrom, to, default,
                            binding.Class);
                    }
                }

                unevaluable += RunCustomPeer(window, binding, pipeline, recentFrom, to, recent, peerTrace) ? 0 : 1;
                RunCustomTrend(window, binding, times, pipeline, from, to, trendTrace);
            }

            return blind;
        }

        /// <summary>
        /// The row for a custom channel that carries a rule and that no pod reported this cycle.
        ///
        /// <para><b>An absent row and an empty trace have to mean different things, or the diagnostic has the
        /// disease it exists to diagnose.</b> After <c>AN-D14</c> a rule trace is empty only when no rule is
        /// configured on any channel — so every OTHER way a configured rule can fall silent needs a row, and
        /// a channel nobody reported is the way that survives the obvious fix.</para>
        ///
        /// <para>The status is <c>InsufficientData</c> and never <c>Healthy</c>: no observation is silence
        /// about the channel, not compliance with the threshold. The gates are still carried, because what an
        /// operator wants next is what the rule WOULD have tested against.</para>
        ///
        /// <para><b>The series is deliberately not read.</b> <c>MetricWindow.Series(pod, name)</c> throws for
        /// a channel the window does not carry, and a binding configured against a channel this cluster never
        /// exports is exactly that case — evaluating the rule "anyway" for symmetry with the built-in path
        /// would turn a silent channel into a failed cycle.</para>
        /// </summary>
        private static void TraceUnreportedRule(
            MetricWindow window,
            in CustomMetricBinding binding,
            Action<RuleDecisionTrace>? ruleTrace)
        {
            if (ruleTrace == null || binding.Rule is not { } rule)
            {
                return;
            }

            for (var pod = 0; pod < window.Pods.Count; pod++)
            {
                ruleTrace(new RuleDecisionTrace(
                    binding.Name,
                    window.Pods[pod],
                    DetectionStatus.InsufficientData,
                    rule.Threshold,
                    rule.MinBreachFraction,
                    0.0,
                    0,
                    0,
                    double.NaN,
                    double.NaN,
                    "No pod reported this channel this cycle, so the rule did not run. Silence about the "
                    + "channel rather than compliance with the threshold."));
            }
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

        /// <inheritdoc cref="GapFloor"/>
        /// <remarks>
        /// <b>Three sources, in this order, and the order is the compatibility contract.</b> The binding's
        /// own step floor wins; absent, its trend floor — which is what this gate used before
        /// <c>MinAbsoluteStepChange</c> existed, so adding the field moved no configured channel; absent
        /// both, the calibrator's step distribution.
        ///
        /// <para>The note that used to sit here said a separate field "would move every existing custom
        /// channel onto the calibrator overnight". That was right about the hazard and wrong about the
        /// remedy: falling back to the trend floor keeps every deployed channel exactly where it was.
        /// Measured on the built-in side first (<c>AN-D4b</c>): sharing the two made the step gate demand
        /// 40% of the level on MemoryWorkingSetBytes and 123% at the low decile of GcGen2HeapBytes, because
        /// a trend floor is fitted to how far ONE series travels across a window and a step floor to how far
        /// the median across pods moves between its halves.</para>
        /// </remarks>
        private double LevelShiftFloor(in CustomMetricBinding binding)
        {
            if (binding.MinAbsoluteStepChange > 0.0)
            {
                return binding.MinAbsoluteStepChange;
            }

            return binding.MinAbsoluteTrendChange > 0.0
                ? binding.MinAbsoluteTrendChange
                : _floors.MinAbsoluteLevelShift(binding.Name);
        }

        /// <returns>Whether the group reached a verdict; false means nobody was compared at all.</returns>
        private bool RunCustomPeer(
            MetricWindow window,
            CustomMetricBinding binding,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to,
            int recent,
            Action<PeerDecisionTrace>? peerTrace = null)
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

            // The gate belongs on BOTH peer call sites. Present only on the built-in one, every channel a
            // customer adds — and on the deployed lab config that is five of them — would keep re-reporting a
            // standing outlier for ever, in exactly the half of the system a client is most likely to extend.
            var pods = new string[podCount];

            for (var pod = 0; pod < podCount; pod++)
            {
                pods[pod] = window.Pods[pod];
            }

            var decisions = Classify(
                pods, findings, result, MetricIndex.Count, binding.Name, binding.MinAbsoluteGapChange, to);

            // Emitted at the same point as RunPeer's — before the demotion — so a row means the same thing on
            // both paths: what the COMPARISON found, with the gates as separate columns beside it.
            //
            // Absent here until 2026-08-10, and the asymmetry cost a measurement the same evening: the trace
            // was released behind a flag to answer "which gate silenced this channel", pointed at a CUSTOM
            // channel, and produced zero rows — because only RunPeer emitted. That is XC-11's shape exactly,
            // one layer along: a diagnostic keyed to the built-in half, missing the half a client extends,
            // which on the deployed lab is six channels.
            //
            // NOT yet shown: HoldUntilPersistent, a gate custom channels have and built-ins do not. A row
            // whose Forwarded is true can still be held back by it one line below.
            if (peerTrace != null)
            {
                for (var i = 0; i < podCount; i++)
                {
                    peerTrace(new PeerDecisionTrace(
                        binding.Name,
                        result.Status,
                        result.HighCount,
                        result.LowCount,
                        pods[i],
                        findings[i].IsOutlier,
                        findings[i].RelativeGap,
                        findings[i].AbsoluteGap,
                        findings[i].Comparison.EffectSize,
                        findings[i].Comparison.PValueCandidateWorse,
                        findings[i].UsableSamples,
                        result.ExcludedCount,
                        decisions == null ? NoveltyKind.New : decisions[i].Kind,
                        decisions == null ? DetectionStatus.InsufficientData : decisions[i].Status,
                        decisions == null || decisions[i].Forward || !findings[i].IsOutlier));
                }
            }

            var kinds = Demote(findings, decisions);

            HoldUntilPersistent(binding, pods, findings, result);

            pipeline.ObservePeerGroup(
                binding.Name, result, findings, subjects, from, to, binding.Class, kinds);

            return result.Status != DetectionStatus.InsufficientData;
        }

        /// <summary>
        /// Clears any deviation on a pod that has not been an outlier for
        /// <see cref="AnomalyGuardOptions.SilentPodCycles"/> consecutive cycles on this channel, so the same
        /// demotion <see cref="Demote"/> performs holds it back until it recurs.
        ///
        /// <para><b>Off unless the channel asks for it, and doing nothing is the default that matters.</b>
        /// <see cref="CustomMetricBinding.RequirePersistence"/> is false for every channel that predates
        /// this, so the five already on this path — <c>CpuPressure</c>, <c>LockContentions</c>,
        /// <c>Exceptions</c>, <c>ActiveRequests</c>, <c>GcCommittedBytes</c> — return here immediately and
        /// report exactly as they did.</para>
        ///
        /// <para><b>Why a channel would ask.</b> Scrape coverage dips for a moment on every pod that is
        /// replaced, and a rollout replaces all of them; a same-cycle gate would therefore add a false
        /// positive per replica per rollout to an incident rate that already fails <c>AN-A1</c>. The dip that
        /// matters is the one still there next cycle.</para>
        ///
        /// <para><b>A cycle that reached no verdict is not evidence in either direction</b>, so it neither
        /// advances nor clears a counter — the same reasoning that makes <see cref="RunSilentPods"/> drop its
        /// counters rather than trust a roster nobody could confirm.</para>
        /// </summary>
        private void HoldUntilPersistent(
            in CustomMetricBinding binding,
            string[] pods,
            PeerOutlierFinding[] findings,
            in PeerOutlierResult result)
        {
            if (!binding.RequirePersistence || _options.SilentPodCycles <= 1)
            {
                return;
            }

            if (result.Status == DetectionStatus.InsufficientData)
            {
                return;
            }

            if (!_customBreach.TryGetValue(binding.Name, out var breaches))
            {
                breaches = new Dictionary<string, int>(pods.Length, StringComparer.Ordinal);
                _customBreach[binding.Name] = breaches;
            }

            for (var i = 0; i < pods.Length; i++)
            {
                if (!findings[i].IsOutlier)
                {
                    // Matched its peers this cycle, which is evidence against the previous one meaning
                    // anything. Removed rather than zeroed so the map does not grow one entry per pod that
                    // has never deviated.
                    breaches.Remove(pods[i]);

                    continue;
                }

                var cycles = breaches.GetValueOrDefault(pods[i]) + 1;
                breaches[pods[i]] = cycles;

                if (cycles >= _options.SilentPodCycles)
                {
                    continue;
                }

                findings[i] = findings[i] with
                {
                    Deviation = PeerDeviation.None
                };
            }

            // Bounded the same way RunSilentPods bounds its own counters, and with the same accepted cost: a
            // pod that leaves the window and comes back re-earns its cycles. It can only leave by reporting
            // nothing at all, which is the silent-pod check's case rather than this one.
            if (breaches.Count <= pods.Length)
            {
                return;
            }

            var stale = new List<string>();

            foreach (var pod in breaches.Keys)
            {
                if (!Contains(pods, pod))
                {
                    stale.Add(pod);
                }
            }

            for (var i = 0; i < stale.Count; i++)
            {
                breaches.Remove(stale[i]);
            }
        }

        private void RunCustomTrend(
            MetricWindow window,
            CustomMetricBinding binding,
            double[] times,
            IncidentPipeline pipeline,
            DateTimeOffset from,
            DateTimeOffset to,
            Action<TrendDecisionTrace>? trendTrace = null)
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
                        pipeline, binding.Name, common, from, to, LevelShiftFloor(binding), binding.Class);
                }
            }

            for (var pod = 0; pod < podCount; pod++)
            {
                // The same warm-up grace RunTrend applies, and its absence here was an omission rather than a
                // decision — added 2026-08-10 (XC-9). The measurement behind WarmUpGrace does not care which
                // enum a channel is keyed by: a fresh replica's series climbs 13-17% of typical over its
                // first 10-20 minutes at a Kendall tau of 0.70-0.94, which is a textbook trend about nothing,
                // and a floor large enough to swallow it is large enough to swallow a real leak.
                //
                // The peer families deliberately do NOT get this — see WarmUpGrace's own remarks: a replica
                // differing from its peers right now is worth reporting whatever its age, and during a
                // rollout every pod is young, so a peer-wide grace would blind the guard exactly when a bad
                // version is going out. This is the trend family, which reads a history, and that is the
                // whole distinction.
                if (IsWarmingUp(window.Pods[pod], to))
                {
                    trendTrace?.Invoke(new TrendDecisionTrace(
                        binding.Name, window.Pods[pod], DetectionStatus.InsufficientData,
                        true, trendFloor, 0, 0, 0, 0, 0,
                        !expectation.IsEmpty, "inside the warm-up grace; no trend test ran"));

                    continue;
                }

                // Custom channels carry their own ceiling on the binding, since the per-metric table is
                // indexed by MetricIndex and cannot hold a name the enum does not have.
                var verdict = _trend.Detect(
                    window.Series(pod, binding.Name),
                    times,
                    options,
                    binding.SaturationLimit,
                    expectation);

                trendTrace?.Invoke(new TrendDecisionTrace(
                    binding.Name, window.Pods[pod], verdict.Status, false, trendFloor,
                    verdict.SlopePerSecond, verdict.KendallTau, verdict.PValue,
                    verdict.Autocorrelation, verdict.SampleCount,
                    !expectation.IsEmpty, verdict.Reason));

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
            int recent,
            Action<RuleDecisionTrace>? ruleTrace = null)
        {
            for (var r = 0; r < _options.Rules.Count; r++)
            {
                var profile = _options.Rules[r];

                for (var pod = 0; pod < window.Pods.Count; pod++)
                {
                    var verdict = _rule.Evaluate(
                        Tail(window.Series(pod, profile.Metric), recent), profile.Options);

                    ruleTrace?.Invoke(new RuleDecisionTrace(
                        profile.Metric.ToString(),
                        window.Pods[pod],
                        verdict.Status,
                        profile.Options.Threshold,
                        profile.Options.MinBreachFraction,
                        verdict.BreachFraction,
                        verdict.BreachedSamples,
                        verdict.UsableSamples,
                        verdict.PeakValue,
                        verdict.MedianValue,
                        verdict.Reason));

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

                var pods = new string[cohort.Count];

                for (var i = 0; i < cohort.Count; i++)
                {
                    pods[i] = window.Pods[cohort[i]];
                }

                var decisions = Classify(
                    pods,
                    findings,
                    result,
                    metric,
                    null,
                    AnomalyGuardOptions.FloorFor(_options.MinAbsoluteGapChange, metric),
                    to);

                // BEFORE the demotion below, because the trace's job is to say what the comparison found
                // and the gate is a separate column on the same row — see PeerDecisionTrace.Novelty.
                if (peerTrace != null)
                {
                    for (var i = 0; i < cohort.Count; i++)
                    {
                        peerTrace(new PeerDecisionTrace(
                            metric.ToString(),
                            result.Status,
                            result.HighCount,
                            result.LowCount,
                            pods[i],
                            findings[i].IsOutlier,
                            findings[i].RelativeGap,
                            findings[i].AbsoluteGap,
                            findings[i].Comparison.EffectSize,
                            findings[i].Comparison.PValueCandidateWorse,
                            findings[i].UsableSamples,
                            result.ExcludedCount,
                            decisions == null ? NoveltyKind.New : decisions[i].Kind,
                            decisions == null ? DetectionStatus.InsufficientData : decisions[i].Status,
                            decisions == null || decisions[i].Forward || !findings[i].IsOutlier));
                    }
                }

                var kinds = Demote(findings, decisions);

                pipeline.ObservePeerGroup(
                    metric.ToString(), result, findings, cohortSubjects, from, to, null, kinds);
            }

            return decided;
        }

        /// <summary>
        /// Runs the novelty gate over one peer group, folding each outlier's gap into that pod's history.
        ///
        /// <para><b>Only outliers are folded, and that is what makes the mechanism robust to a noisy
        /// ranking.</b> A pod whose gap sits under the material floor never reaches the tracker, so the
        /// role-churn measured on the live population — the top-ranked pod changing 17 times in 180
        /// transitions, almost all of it below the floor — never becomes novelty history about anybody.</para>
        ///
        /// <para>Returns null when the gate is off, which every caller reads as "forward everything".</para>
        /// </summary>
        private NoveltyDecision[]? Classify(
            string[] pods,
            PeerOutlierFinding[] findings,
            in PeerOutlierResult result,
            MetricIndex metric,
            string? channel,
            double minAbsoluteGapChange,
            DateTimeOffset at)
        {
            if (_novelty == null || result.Status != DetectionStatus.Anomalous)
            {
                return null;
            }

            var decisions = new NoveltyDecision[findings.Length];

            for (var i = 0; i < findings.Length; i++)
            {
                if (!findings[i].IsOutlier)
                {
                    // Not folded and not judged: a cycle in which this pod matched its peers says nothing
                    // about whether its deviations are standing, and recording a zero gap for it would tell
                    // the change test the gap collapsed.
                    decisions[i] = NoveltyDecision.Unknown;

                    continue;
                }

                var createdAt = CreatedAt(pods[i]);

                decisions[i] = channel == null
                    ? _novelty.Observe(
                        pods[i], createdAt, metric, findings[i].AbsoluteGap, at, minAbsoluteGapChange)
                    : _novelty.Observe(
                        pods[i], createdAt, channel, findings[i].AbsoluteGap, at, minAbsoluteGapChange);
            }

            return decisions;
        }

        /// <summary>
        /// Clears the deviation on every member the gate held back this cycle, so
        /// <c>IncidentPipeline.ObservePeerGroup</c>'s existing outlier check skips it — the same demotion
        /// <c>PeerGroupOutlierDetector.Dominant</c> already performs, one gate further out.
        /// </summary>
        /// <returns>Per-member classification for the pipeline, or null when the gate is off.</returns>
        private NoveltyKind[]? Demote(PeerOutlierFinding[] findings, NoveltyDecision[]? decisions)
        {
            if (decisions == null)
            {
                return null;
            }

            var kinds = new NoveltyKind[decisions.Length];

            for (var i = 0; i < decisions.Length; i++)
            {
                kinds[i] = decisions[i].Kind;

                if (decisions[i].Forward || !findings[i].IsOutlier)
                {
                    continue;
                }

                findings[i] = findings[i] with
                {
                    Deviation = PeerDeviation.None
                };

                Telemetry.NoveltyHeld();
            }

            return kinds;
        }

        /// <summary>
        /// What the cluster says about when a pod was created, or <c>default</c> when it cannot say. See
        /// <c>PeerNoveltyTracker.Observe</c> for why a change in this value discards that pod's history.
        /// </summary>
        private DateTimeOffset CreatedAt(string pod)
        {
            if (_options.PodTopology is { } topology && topology.TryResolve(pod, out var placement))
            {
                return placement.CreatedAt;
            }

            return default;
        }

        /// <summary>
        /// Drops novelty state for pods the cluster no longer lists.
        ///
        /// <para><b>Against the roster, never against the reporting window.</b> A pod that missed one scrape
        /// is absent from the window and still very much alive; pruning on that would discard its history and
        /// silently restart its warm-up. A stale roster can only cause a pod to be forgotten early, which
        /// makes it report again — the safe direction for a suppression gate.</para>
        /// </summary>
        private void PruneNovelty()
        {
            if (_novelty == null || _options.PodTopology is not IPodRoster roster)
            {
                return;
            }

            var known = roster.KnownPods;

            if (known.Count == 0)
            {
                return;
            }

            _novelty.Prune(known);
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
            DateTimeOffset to,
            Action<TrendDecisionTrace>? trendTrace = null)
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
                    // The cross-peer component wins for the PER-POD trends, and the priority used to be the
                    // other way round. That is `AN-F1`, and it cost 11 -> 33 opened incidents on an
                    // identical window.
                    //
                    // The two references answer different questions. `common` removes what the replicas are
                    // doing together RIGHT NOW; `seasonal` removes what this workload usually does at this
                    // hour. Judging a pod against the seasonal expectation therefore leaves every
                    // common-mode movement inside its own series, so a fleet climbing together produces one
                    // finding per pod. Measured 2026-08-10 at twelve pods: a +15% shared climb gives 0
                    // findings against `common` and 12 against `seasonal`, while a flat fleet gives 0 for
                    // both — the arm that would have refuted the explanation.
                    //
                    // They are not composed, and that was the wrong first instinct: both are point estimates
                    // in the metric's own units, so adding them would double the signal's magnitude rather
                    // than remove two effects.
                    //
                    // `RunCustomTrend` already had exactly this shape (`expectation = common` on success),
                    // so this makes two structurally parallel methods agree rather than inventing a policy.
                    // The WORKLOAD-level arm below is unchanged: judging the common component against the
                    // seasonal expectation is the composition done correctly, one level up.
                    expectation = common;

                    var verdict = _trend.Detect(common, times, options, double.NaN, seasonal);

                    pipeline.Observe(WorkloadSubject(), metric.ToString(), verdict, from, to, common);

                    // Its OWN floor, not the trend one. Sharing them put the CPU gate at 0.81 cores against
                    // a real 0.39-core step and made the only fault this family can see unreportable.
                    ObserveLevelShift(
                        pipeline, metric.ToString(), Adjust(common, seasonal), from, to,
                        _floors.MinAbsoluteLevelShift(metric), null);

                }
            }

            // Learned OUTSIDE the decomposition branch, and that placement is the fix rather than a detail.
            // It used to sit inside, so turning DecomposeCommonMode off silently disabled a week of seasonal
            // learning — one option switching off an unrelated subsystem as a side effect nobody would
            // predict from its name. The level comes from the common component when there is one and from
            // the pods' own medians when there is not; the two are close, and a slightly coarser baseline
            // beats no baseline by a distance.
            if (!_declaredAbnormal && _history != null)
            {
                var level = common != null ? Median(common) : MedianAcrossPods(window, metric);

                _history.Observe(_workload, metric, from, level);
            }

            // Which reference the per-pod trends are about to use, made visible. Without this the fallback
            // to seasonal is silent, and its consequence — one shared climb arriving as one finding per pod —
            // looks like a fleet-wide fault rather than like the regime the guard is in.
            if (common == null && !expectation.IsEmpty)
            {
                Telemetry.TrendSeasonalOnly();
            }

            for (var pod = 0; pod < podCount; pod++)
            {
                if (IsWarmingUp(window.Pods[pod], to))
                {
                    // Traced before the skip. A bare `continue` leaves "no finding" and "never tested"
                    // looking identical, and during a rollout that is every pod at once.
                    trendTrace?.Invoke(new TrendDecisionTrace(
                        metric.ToString(), window.Pods[pod], DetectionStatus.InsufficientData,
                        true, options.MinAbsoluteChangeOverWindow, 0, 0, 0, 0, 0,
                        !expectation.IsEmpty, "inside the warm-up grace; no trend test ran"));

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

                trendTrace?.Invoke(new TrendDecisionTrace(
                    metric.ToString(), window.Pods[pod], verdict.Status, false,
                    options.MinAbsoluteChangeOverWindow, verdict.SlopePerSecond, verdict.KendallTau,
                    verdict.PValue, verdict.Autocorrelation, verdict.SampleCount,
                    !expectation.IsEmpty, verdict.Reason));

                pipeline.Observe(
                    Subject(window.Pods[pod]), metric.ToString(), verdict, from, to,
                    verdict.Status == DetectionStatus.Anomalous
                        ? window.Series(pod, metric).ToArray()
                        : default);
            }
        }

        /// <summary>
        /// The seasonal expectation for this window, or an EMPTY span when there is not enough history to
        /// have one — the caller treats empty as "no expectation" and falls back to the cross-peer
        /// common component.
        ///
        /// <para>The previous sentence here named a <c>fallback</c> parameter this method does not take
        /// and has not taken since it started returning an empty span instead. Recorded rather than
        /// silently corrected: prose drifting away from a signature is the defect class this codebase
        /// keeps finding by reading, and the compiler was reporting it as CS1734 the whole time.</para>
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
            if (_history == null || _options.MinimumHistoryDays <= 0)
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
            => samples >= series.Length ? series : series.Slice(series.Length - samples);

        /// <summary>
        /// The last <paramref name="samples"/> of a signal, as memory over the window's own storage.
        ///
        /// <para>The peer detector reads its input inside the call and its findings carry numbers rather than
        /// series, so nothing here outlives the window — which is what makes a slice safe where the trend
        /// path still has to copy. It was the last per-pod allocation in a cycle: two copies of the tail per
        /// replica per signal, made whether or not anything came of them.</para>
        /// </summary>
        private static ReadOnlyMemory<double> TailMemory(ReadOnlyMemory<double> series, int samples)
            => samples >= series.Length ? series : series.Slice(series.Length - samples);

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
                    return pod.Substring(0, secondLast);
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
