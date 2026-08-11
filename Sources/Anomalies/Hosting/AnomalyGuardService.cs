// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Runtime;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Anomalies.Hosting
{
    /// <summary>
    /// The loop: read a window, evaluate it, report what changed, wait, repeat.
    ///
    /// <para>Everything it does lives elsewhere — an <see cref="IMetricWindowSource"/> fetches,
    /// <see cref="AnomalyGuard"/> evaluates, an <see cref="IIncidentSink"/> reports. This is scheduling and
    /// failure handling, which is why it is the only part that belongs to the host rather than the library.</para>
    ///
    /// <para><b>The window ends behind "now" on purpose.</b> Rate expressions are computed over a trailing
    /// range, so samples at the current instant are still filling in. Measured on the lab: a window ending at
    /// the moment a load run stopped put a cluster-wide downward trend in every RED signal, entirely an
    /// artefact of when the measurement ended.</para>
    ///
    /// <para><b>A failed cycle is logged and skipped, never fatal.</b> A monitoring guard that dies because
    /// Prometheus was briefly unreachable has replaced the problem it was bought to detect with one of its
    /// own — and it takes the host down with it.</para>
    ///
    /// <para><b>It also reports when it cannot see.</b> A metric no pod exports produces no findings, which is
    /// indistinguishable from health at every layer below. <see cref="GuardCycleResult.BlindMetrics"/> is
    /// logged as its own event so "quiet" and "blind" are separable in a query, which they are not if the
    /// only evidence for either is an absence of incident lines.</para>
    /// </summary>
    public sealed class AnomalyGuardService : BackgroundService
    {
        private const int BlindEventId = 5004;
        private const int BlindMetricEventId = 5007;
        private const int CycleFailedEventId = 5005;
        private const int RestoredEventId = 5008;
        private const int CycleEventId = 5009;
        private const int FloorProposalEventId = 5010;
        private const int TopologyStaleEventId = 5006;
        private const int PeerTraceEventId = 5011;
        private const int TrendTraceEventId = 5012;
        private const int RuleTraceEventId = 5013;

        private static readonly Action<ILogger, int, int, int, int, Exception?> _blind =
            LoggerMessage.Define<int, int, int, int>(
                LogLevel.Warning,
                new EventId(BlindEventId, "AnomalyGuardPartiallyBlind"),
                "Anomaly guard is partly blind: {BlindMetrics} metric(s) returned nothing for any pod and "
                + "{PartialMetrics} for only some, across {Pods} pods — an absence of incidents this cycle "
                + "({Incidents}) does not mean the cluster is healthy");

        // One line per blind metric, alongside the count above. The count says the guard is partly blind; the
        // name says which query to go and fix, and that is the whole actionable part.
        private static readonly Action<ILogger, string, Exception?> _blindMetric =
            LoggerMessage.Define<string>(
                LogLevel.Warning,
                new EventId(BlindMetricEventId, nameof(BlindMetricEventId)),
                "Blind on {Metric}: no pod reported it this cycle. Either this cluster does not export it "
                + "or its binding is wrong — both look like health from here.");

        // Its own event, because stale topology degrades grouping quietly rather than loudly: the guard keeps
        // producing incidents, they are just related to each other by out-of-date coordinates.
        private static readonly Action<ILogger, int, Exception?> _topologyStale =
            LoggerMessage.Define<int>(
                LogLevel.Warning,
                new EventId(TopologyStaleEventId, "AnomalyGuardTopologyStale"),
                "Pod topology could not be refreshed; grouping this cycle used the previous snapshot of "
                + "{Pods} pod(s). A pod created since then falls back to a name heuristic.");

        // One line per cycle, always, including the quiet ones. Counting incidents over a day is the number
        // that decides whether anyone can page on this, and it cannot be recovered from incident lines alone:
        // an absence of them is ambiguous between "nothing happened", "the guard was blind" and "the guard
        // was not running". A summary emitted every cycle makes the denominator explicit.
        private static readonly EventId CycleEvent = new(CycleEventId, "AnomalyGuardCycle");
        private static readonly EventId PeerTraceEvent = new(PeerTraceEventId, "AnomalyGuardPeerTrace");
        private static readonly EventId TrendTraceEvent = new(TrendTraceEventId, "AnomalyGuardTrendTrace");
        private static readonly EventId RuleTraceEvent = new(RuleTraceEventId, "AnomalyGuardRuleTrace");

        private static readonly Action<ILogger, int, Exception?> _restored =
            LoggerMessage.Define<int>(
                LogLevel.Information,
                new EventId(RestoredEventId, "AnomalyGuardStateRestored"),
                "Adopted {Restored} open incident(s) from durable state. Zero after a restart that should "
                + "have restored something means the state did not survive, and the next cycle will report "
                + "problems the operator was already told about as new.");

        // Deliberately Warning, not Information. A configured floor sitting below what the cluster does when
        // nothing is wrong is not a curiosity — it is the guard telling the operator, with evidence, which of
        // its own settings is generating noise.
        private static readonly Action<ILogger, string, double, double, double, int, Exception?> _floorProposal =
            LoggerMessage.Define<string, double, double, double, int>(
                LogLevel.Warning,
                new EventId(FloorProposalEventId, "AnomalyGuardFloorProposal"),
                "Floor proposal for {Metric}: healthy peers differed by up to {ObservedMax} (typical "
                + "magnitude {Typical}), so a gap below {Proposed} is something this cluster does when it is "
                + "well. The configured floor is under that, which is why it reports. Based on {Samples} "
                + "observation(s) — valid only if this period really was healthy.");

        // Observations and windows are reported separately because one does not imply the other and the
        // difference has already cost a false positive: twelve replicas in one window are twelve
        // observations of a single moment. "Based on 480 observations" reads as a lot and can be forty
        // windows or four.
        private static readonly Action<ILogger, string, int, int, Exception?> _floorTooEarly =
            LoggerMessage.Define<string, int, int>(
                LogLevel.Information,
                new EventId(FloorProposalEventId, "AnomalyGuardFloorProposalTooEarly"),
                "No floor proposal for {Metric} yet: {Samples} observation(s) across {Windows} window(s). A "
                + "floor set from too few windows describes the last rollout rather than the workload — "
                + "measured 2026-08-09, where three minutes gave half the peak that twenty minutes did.");

        private static readonly Action<ILogger, Exception?> _cycleFailed =
            LoggerMessage.Define(
                LogLevel.Error,
                new EventId(CycleFailedEventId, "AnomalyGuardCycleFailed"),
                "Anomaly guard cycle failed; skipping it and continuing.");

        /// <summary>
        /// Channel filter for the peer-decision trace, from <see cref="OverfitEnvironment.GuardPeerTrace"/>.
        /// Null when the trace is off, empty when every channel is traced, otherwise the one channel asked for.
        ///
        /// <para>Read once, at construction: a flag re-read every cycle is a flag whose value nobody can state
        /// while reading a log.</para>
        /// </summary>
        private readonly string? _peerTraceFilter;

        private readonly IClock _clock;
        private readonly AnomalyGuardServiceOptions _options;
        private readonly IMetricWindowSource _source;
        private readonly IRefreshablePodTopology? _topology;
        private readonly AnomalyGuard _guard;
        private readonly ILogger<AnomalyGuardService> _logger;
        private readonly FloorCalibrator? _calibrator;

        /// <summary>
        /// Features this deployment has no binding for, so their per-cycle silence is expected.
        ///
        /// <para><b>Two different things were being reported as one, and the harmless one was burying the
        /// dangerous one.</b> A metric with no binding is a configuration fact: known before the first cycle,
        /// unchanging, and already stated once at startup. A metric that IS bound and that no pod reported is
        /// a broken exporter or a severed scrape — the failure this whole subsystem exists to surface, and it
        /// needs an operator today. Both produced the same warning on every cycle, so on the lab a
        /// permanently unbound <c>CpuThrottleRatio</c> raised one 292 times in 292 cycles, and any real
        /// exporter failure would have arrived looking exactly like the noise everybody had learned to
        /// ignore.</para>
        ///
        /// <para>The counted form is unchanged: <c>blind=N</c> on the cycle line still includes these, because
        /// they are genuinely blind spots. What is reserved for the actionable case is the WARNING.</para>
        /// </summary>
        private readonly bool[] _unbound = new bool[(int)MetricIndex.Count];
        private int _knownPods;
        private DateTimeOffset _nextProposal;

        /// <param name="options">Cadence, window and the guard's own thresholds; the defaults are the
        /// measured ones and the window especially should not be widened on intuition — a four-hour window
        /// sits on the slope of the daily traffic curve and measured 2583 false incidents a day against 234
        /// at twenty minutes.</param>
        /// <param name="source">Reads the window this cycle evaluates. One instance for the process; the
        /// live implementation lends its HTTP client to a per-cycle Prometheus source. Taken as the interface
        /// rather than as <see cref="PrometheusMetricWindowSource"/> so a window that has already passed can
        /// drive the same loop — which is what makes a threshold change testable against a fixed body of
        /// history instead of against another day of cluster time.</param>
        /// <param name="sink">Where findings and incidents go. Defaults to the logging sink in shadow mode,
        /// which counts and explains and wakes nobody.</param>
        /// <param name="logger">The loop's own voice. Everything an operator learns about coverage — blind
        /// channels, excluded pods, unwritable state — arrives here rather than as a metric, because names
        /// are the actionable part and a counter has none.</param>
        /// <param name="topology">
        /// Pod ownership, refreshed before each window so grouping relates findings by the cluster as it is
        /// now. Optional, and its absence is not neutral: without it the grouper falls back to a name
        /// heuristic and any pod it guesses wrong is merged into the wrong incident.
        /// </param>
        /// <param name="store">
        /// Optional durable state. <b>Supply one in any deployment that can be restarted</b>, which is all of
        /// them: without it every incident that was running is reopened after a rollout or a crash, and the
        /// operator is paged again for problems they were already told about — the tracker's whole
        /// contribution undone by the guard's own restart. This was missing here while
        /// <see cref="AnomalyGuard"/> had supported it all along, so the durable path existed and nothing
        /// deployable reached it.
        /// </param>
        /// <param name="learnedState">
        /// Where the seasonal baseline and the calibrated floors live. Separate from <paramref name="store"/>
        /// because the two fail differently: losing the incidents costs one burst of duplicate notifications,
        /// losing this costs a week of relearning during which the guard is QUIETER than it should be — the
        /// failure that looks like success.
        /// </param>
        /// <param name="metricMap">
        /// The configured bindings, used only to tell an unbound channel from a bound one that reported
        /// nothing. Optional, and omitting it is the conservative choice: every silent metric is then warned
        /// about every cycle, which says too much rather than too little.
        /// </param>
        public AnomalyGuardService(
            AnomalyGuardServiceOptions options,
            IMetricWindowSource source,
            IIncidentSink sink,
            ILogger<AnomalyGuardService> logger,
            IRefreshablePodTopology? topology = null,
            IIncidentStore? store = null,
            ILearnedStateStore? learnedState = null,
            MetricMap? metricMap = null,
            IClock? clock = null)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(source);
            ArgumentNullException.ThrowIfNull(sink);
            ArgumentNullException.ThrowIfNull(logger);

            _clock = clock ?? SystemClock.Instance;
            _peerTraceFilter = ReadPeerTraceFilter();

            _options = options;
            _source = source;
            _topology = topology;
            _logger = logger;

            // Optional, and the default is the conservative one: with no map every silent metric is treated
            // as bound-and-silent, which warns too much rather than too little. A guard that has lost track
            // of its own configuration should err towards saying something.
            if (metricMap != null)
            {
                var unmapped = metricMap.Unmapped;

                for (var i = 0; i < unmapped.Count; i++)
                {
                    _unbound[(int)unmapped[i]] = true;
                }
            }

            if (options.FloorProposalInterval > TimeSpan.Zero)
            {
                _calibrator = new FloorCalibrator(options.Guard.Trend);
            }

            // The guard reads topology through the interface, so handing it the same instance the loop
            // refreshes is what keeps the two in step — no snapshot is copied anywhere.
            _guard = new AnomalyGuard(
                options.Guard with
                {
                    PodTopology = topology
                },
                sink,
                options.Tracking,
                store,
                restoredAt: null,

                // Passed, and it had better stay passed. The same omission on the incident store meant
                // durable state existed in the library while nothing deployable reached it — the seasonal
                // baseline and the floor calibration would have relearned from nothing on every rollout, and
                // a guard that has forgotten its floors is quietly the noisy one.
                historyStore: learnedState);
        }

        /// <summary>Incidents adopted from durable state when this instance started. Zero without a store.</summary>
        public int RestoredIncidents => _guard.RestoredIncidents;

        /// <summary>
        /// The guard's own counters, for a host to expose to Prometheus. <b>Alert on
        /// <c>overfit_guard_last_cycle_timestamp_seconds</c> with <c>absent()</c> as well as a staleness
        /// comparison</b>: the comparison alone cannot fire once the pod is gone, since the series is gone
        /// too. Measured 2026-08-05; <c>k8s/lab/guard-alerts.yaml</c> carries the working rule. A stopped
        /// guard is worse than an absent one because somebody is relying on it.
        /// </summary>
        public GuardTelemetry Telemetry => _guard.Telemetry;

        /// <summary>
        /// The guard itself, so a host can route an operator's acknowledgement to it.
        ///
        /// <para><b>Through the running process, not through the state file.</b> The guard rewrites its
        /// learned state every cycle from memory, so a CLI editing that file would have its work overwritten
        /// within one cadence — silently, with no error anywhere. That is why acknowledging is an HTTP call
        /// to this process rather than the obvious file edit; the obvious version loses data and looks like
        /// it worked.</para>
        ///
        /// <para>Its acknowledgement methods take the cycle lock, so calling them from a request thread is
        /// safe while the evaluation loop runs.</para>
        /// </summary>
        public AnomalyGuard Guard => _guard;

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            // Said once, at startup, because it is the only evidence an operator has that durable state is
            // wired at all. Zero after a restart that should have restored something means either no store
            // was supplied or the file did not survive — and both present as a burst of duplicate
            // notifications for problems the operator was already told about.
            _restored(_logger, _guard.RestoredIncidents, null);

            using var timer = new PeriodicTimer(_options.Cadence);

            // #pragma BOUND: exits when the host signals cancellation; PeriodicTimer.WaitForNextTickAsync
            // returns false once the token fires, so there is no other way out and no way to spin.
            while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
            {
                // The one wall-clock read in this subsystem's cycle path, and it lives here rather than
                // inside the cycle so a caller replaying history can name a different moment. Live behaviour
                // is the same read, one frame up.
                await RunCycleAsync(_clock.UtcNow, stoppingToken).ConfigureAwait(false);
            }
        }

        /// <summary>
        /// One line per trend decision. Carries the detector's own <c>Reason</c> sentence alongside the
        /// numbers behind it, because that sentence is what an operator reads first and the numbers are what
        /// they check it against.
        ///
        /// <para><c>warmingUp=True</c> is the row that did not exist before: a pod inside the grace is
        /// skipped with a bare <c>continue</c>, so "tested and healthy" and "never tested" produced the same
        /// silence — and during a rollout that is every pod at once.</para>
        /// </summary>
        private void LogTrendDecision(TrendDecisionTrace row)
        {
            if (_peerTraceFilter is not { } filter)
            {
                return;
            }

            if (filter.Length > 0 && !string.Equals(filter, row.Signal, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            _logger.LogInformation(
                TrendTraceEvent,
                "trend-trace: signal={Signal} pod={Pod} status={Status} warmingUp={WarmingUp} "
                + "floor={FloorOverWindow} slope={SlopePerSecond} tau={KendallTau} p={PValue} "
                + "autocorr={Autocorrelation} samples={SampleCount} seasonal={HasExpectation} "
                + "reason={Reason}",
                row.Signal, row.Pod, row.Status, row.WarmingUp, row.FloorOverWindow, row.SlopePerSecond,
                row.KendallTau, row.PValue, row.Autocorrelation, row.SampleCount, row.HasExpectation,
                row.Reason);
        }

        /// <summary>
        /// One line per absolute-threshold decision. Both gates are separate columns: a window can be over
        /// the threshold and still produce nothing because it was over it for too little of the window, and
        /// those two call for opposite fixes.
        /// </summary>
        private void LogRuleDecision(RuleDecisionTrace row)
        {
            if (_peerTraceFilter is not { } filter)
            {
                return;
            }

            if (filter.Length > 0 && !string.Equals(filter, row.Signal, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            _logger.LogInformation(
                RuleTraceEvent,
                "rule-trace: signal={Signal} pod={Pod} status={Status} threshold={Threshold} "
                + "minBreach={MinBreachFraction} breach={BreachFraction} breached={BreachedSamples} "
                + "samples={UsableSamples} peak={PeakValue} median={MedianValue} reason={Reason}",
                row.Signal, row.Pod, row.Status, row.Threshold, row.MinBreachFraction, row.BreachFraction,
                row.BreachedSamples, row.UsableSamples, row.PeakValue, row.MedianValue, row.Reason);
        }

        /// <summary>
        /// Reads <see cref="OverfitEnvironment.GuardPeerTrace"/> into a filter.
        ///
        /// <para>Null is off. Empty string means every channel. Anything else is a channel name, because the
        /// question this trace answers is almost always about ONE channel — twelve pods times fourteen
        /// channels is 168 rows a cycle, and an operator hunting one silent signal should not have to grep
        /// past the other thirteen.</para>
        /// </summary>
        private static string? ReadPeerTraceFilter()
        {
            var raw = Environment.GetEnvironmentVariable(OverfitEnvironment.GuardPeerTrace)?.Trim();

            if (string.IsNullOrEmpty(raw))
            {
                return null;
            }

            if (string.Equals(raw, "1", StringComparison.Ordinal)
                || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(raw, "all", StringComparison.OrdinalIgnoreCase))
            {
                return string.Empty;
            }

            return raw;
        }

        /// <summary>
        /// One line per peer comparison, carrying every gate separately.
        ///
        /// <para><b>Separately, because a single verdict cannot be acted on.</b> "No finding" has five causes
        /// that call for opposite fixes: the rank test found no consistent difference, the relative gap was
        /// under its gate, the absolute gap was under its floor, the member had too few usable samples, or the
        /// novelty gate suppressed a real one. Lowering a floor when the rank test is what refused would be
        /// the wrong fix applied confidently.</para>
        ///
        /// <para>Logged at Information rather than Debug: the flag is already the gate, and a diagnostic that
        /// needs a second knob turned before it appears is one an operator gives up on.</para>
        /// </summary>
        private void LogPeerDecision(PeerDecisionTrace row)
        {
            if (_peerTraceFilter is not { } filter)
            {
                return;
            }

            if (filter.Length > 0 && !string.Equals(filter, row.Signal, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            // Fourteen fields against LoggerMessage.Define's six, and the same trade-off the cycle line
            // documents: this runs only under an explicit flag, so the allocation is not worth splitting the
            // row for. The placeholders still name the properties a structured sink records.
            _logger.LogInformation(
                PeerTraceEvent,
                "peer-trace: signal={Signal} pod={Pod} status={Status} outlier={IsOutlier} "
                + "relGap={RelativeGap} absGap={AbsoluteGap} effect={EffectSize} p={PValue} "
                + "samples={UsableSamples} excluded={ExcludedPeers} high={High} low={Low} "
                + "novelty={Novelty} noveltyStatus={NoveltyStatus} forwarded={Forwarded}",
                row.Signal,
                row.Pod,
                row.Status,
                row.IsOutlier,
                row.RelativeGap,
                row.AbsoluteGap,
                row.EffectSize,
                row.PValue,
                row.UsableSamples,
                row.ExcludedPeers,
                row.High,
                row.Low,
                row.Novelty,
                row.NoveltyStatus,
                row.Forwarded);
        }

        /// <summary>
        /// Brings the topology snapshot up to date, before the window rather than after: grouping this cycle
        /// should relate findings by the cluster as it is now, and a pod created since the last refresh would
        /// otherwise be related by a stale answer.
        ///
        /// <para>A refusal is reported and the previous snapshot stands. Adopting an empty one would give
        /// every pod the same blank workload, and the grouper would then merge the whole namespace into one
        /// incident — worse than being slightly out of date.</para>
        /// </summary>
        private async Task RefreshTopologyAsync(CancellationToken ct)
        {
            if (_topology == null)
            {
                return;
            }

            var resolved = await _topology.RefreshAsync(ct).ConfigureAwait(false);

            if (resolved < 0)
            {
                _topologyStale(_logger, _knownPods, null);

                return;
            }

            _knownPods = resolved;
        }

        /// <summary>
        /// Folds this window into the running picture of what healthy looks like, and periodically says what
        /// floors that picture implies.
        ///
        /// <para><b>Only the metrics whose configured floor is below the proposal are reported</b>, because
        /// those are the only ones where the setting is the reason for the noise. A floor already above what
        /// the cluster does is doing its job and there is nothing to say about it — so the report shrinks as
        /// the configuration is fixed, instead of restating the same thirteen lines every hour until they are
        /// filtered out and stop being read.</para>
        /// </summary>
        private void ProposeFloors(MetricWindow window, DateTimeOffset now)
        {
            if (_calibrator == null)
            {
                return;
            }

            _calibrator.Observe(window);

            if (_nextProposal == default)
            {
                _nextProposal = now + _options.FloorProposalInterval;

                return;
            }

            if (now < _nextProposal)
            {
                return;
            }

            _nextProposal = now + _options.FloorProposalInterval;

            var proposals = _calibrator.Propose();

            for (var m = 0; m < proposals.Length; m++)
            {
                var proposal = proposals[m];

                var metric = (MetricIndex)m;

                // Said once per proposal interval rather than silently skipped. An operator waiting for a
                // floor and hearing nothing cannot tell "not enough history yet" from "this channel is
                // fine" — and the second is what silence usually means everywhere else in this log.
                if (proposal.Samples > 0 && proposal.Windows < FloorProposal.MinimumWindows)
                {
                    _floorTooEarly(_logger, metric.ToString(), proposal.Samples, proposal.Windows, null);

                    continue;
                }

                // Gated on usability alone. It used to also require ProposedMinAbsoluteGap > 0, which
                // silently skipped every gate on a channel whose PEER proposal happened to be empty — a
                // channel could have a perfectly good step or trend proposal and never be mentioned. Each
                // Propose call already refuses its own empty proposal, so this only has to answer "is there
                // enough history to say anything at all".
                if (!proposal.IsUsable)
                {
                    continue;
                }

                // All THREE gates, because they are read by different families and reporting a subset is how
                // the unreported one goes unexamined for months. The peer gate governs how far apart two
                // replicas may sit; the trend gate how far one may move across a window; the step gate how
                // far the median across pods moves between the halves of one.
                //
                // The step gate was added 2026-08-11 (AN-D4b) and its absence had a measured cost: because
                // the step floor falls back to the trend floor, and a trend floor is fitted to a much larger
                // quantity, the step gate was demanding 40% of the level on MemoryWorkingSetBytes and 123%
                // at the low decile of GcGen2HeapBytes — and the calibrator had been computing the right
                // number the whole time, into a field nothing printed.
                Propose(
                    metric, "peer gap",
                    AnomalyGuardOptions.FloorFor(_options.Guard.MinAbsoluteGap, metric),
                    proposal.PeerGapMax, proposal.ProposedMinAbsoluteGap, proposal);

                Propose(
                    metric, "trend change across a window",
                    AnomalyGuardOptions.FloorFor(_options.Guard.MinAbsoluteTrendChange, metric),
                    proposal.TrendChangeMax, proposal.ProposedMinAbsoluteTrendChange, proposal);

                // Compared against the EFFECTIVE step floor, not against the step table alone. With no
                // minStepChange configured the gate really runs on the trend floor, so comparing against an
                // unset table would print a proposal on every interval for a floor that already covers it —
                // advice that is wrong rather than merely noisy.
                Propose(
                    metric, "step in the workload's level",
                    EffectiveStepFloor(metric),
                    proposal.LevelShiftMax, proposal.ProposedMinAbsoluteLevelShift, proposal);
            }
        }

        /// <summary>
        /// The floor the step gate actually uses: its own table where configured, the trend floor
        /// otherwise. Mirrors <c>ConfiguredFloorSource.MinAbsoluteLevelShift</c> — the two must agree, or
        /// the log proposes against a number the detector does not use.
        /// </summary>
        private double EffectiveStepFloor(MetricIndex metric)
        {
            var step = AnomalyGuardOptions.FloorFor(_options.Guard.MinAbsoluteStepChange, metric);

            return step > 0.0 ? step : AnomalyGuardOptions.FloorFor(_options.Guard.MinAbsoluteTrendChange, metric);
        }

        /// <summary>
        /// Reports one gate's proposal, or says nothing when the configured floor already covers it.
        /// </summary>
        private void Propose(
            MetricIndex metric, string gate, double configured, double observed, double proposed,
            in FloorProposal proposal)
        {
            if (proposed <= 0.0 || configured >= proposed)
            {
                return;
            }

            _floorProposal(
                _logger,
                $"{metric} ({gate})",
                observed,
                proposal.TypicalMagnitude,
                proposed,
                proposal.Samples,
                null);
        }

        /// <summary>
        /// One evaluation cycle, against the moment the caller names rather than against the wall clock.
        ///
        /// <para><b>Taking <paramref name="now"/> as a parameter is what makes a threshold change testable.</b>
        /// The cycle uses it twice — the window it asks the source for ends at <c>now - EndOffset</c>, and the
        /// guard ages its incidents against it — so a method that read the clock itself could only ever
        /// evaluate the present, whatever source it was handed. Re-evaluating a fixed body of history after
        /// each threshold change then costs another day of cluster time instead of a test run.</para>
        ///
        /// <para><b>One ordering difference from the version that read the clock inside.</b> A parameter is
        /// evaluated before the call, so <paramref name="now"/> is read before the topology refresh rather
        /// than after it — earlier by the duration of one refresh. It moves the window end by that much
        /// against an <c>EndOffset</c> whose default is two minutes, which is why this is recorded rather
        /// than guarded.</para>
        ///
        /// <para><b>Deterministic for store-less replay only.</b> Two cold instances driven through the same
        /// windows and the same <paramref name="now"/> sequence produce equal results;
        /// <see cref="AnomalyGuard"/> still falls back to the wall clock for <c>restoredAt</c> when a durable
        /// store is supplied, and this service passes none.</para>
        /// </summary>
        /// <param name="now">The moment this cycle is evaluated as of.</param>
        /// <param name="ct">Cancellation.</param>
        /// <returns>
        /// How the cycle ended, and what it decided when it completed.
        ///
        /// <para><b>Three outcomes rather than a nullable result</b>, because the two absent cases are not the
        /// same finding: a cluster the source cannot see and a cycle that threw are logged as different events
        /// here, and a caller collecting results — which is the reason this method returns anything at all —
        /// could not tell them apart while both were <c>null</c>. See <see cref="GuardCycleOutcome"/> for why
        /// the result is reachable only through <c>TryGetResult</c>.</para>
        /// </returns>
        internal async Task<GuardCycleOutcome> RunCycleAsync(DateTimeOffset now, CancellationToken ct)
        {
            try
            {
                await RefreshTopologyAsync(ct).ConfigureAwait(false);

                var end = now - _options.EndOffset;
                var window = await _source.ReadAsync(end, _options.Window, ct).ConfigureAwait(false);

                if (window == null)
                {
                    // No pod returned anything. Not an empty cluster — a cluster this source cannot see.
                    _blind(_logger, (int)MetricIndex.Count, 0, 0, 0, null);

                    return GuardCycleOutcome.Blind;
                }

                // The trace is built only when the flag is on, so the ordinary path allocates no delegate and
                // the guard takes the null branch it always did. Passing one unconditionally would make every
                // cycle pay for a diagnostic nobody asked for.
                var result = _peerTraceFilter == null
                    ? _guard.RunCycle(window, now)
                    : _guard.RunCycle(
                        window, now, null, LogPeerDecision, LogTrendDecision, LogRuleDecision);

                // Eight fields, and LoggerMessage.Define stops at six. Pre-compiling this would mean dropping
                // two of them or splitting the line, and neither is worth it for a call that happens once per
                // cadence — the allocation is nothing against a cycle that has just read a window from
                // Prometheus. The placeholders still name the properties a structured sink records.
                _logger.LogInformation(
                    CycleEvent,
                    "cycle: pods={Pods} findings={Findings} incidents={Incidents} opened={Opened} "
                    + "ongoing={Ongoing} resolved={Resolved} blind={Blind} unevaluable={Unevaluable}",
                    window.Pods.Count,
                    result.Findings,
                    result.Incidents,
                    result.Opened,
                    result.Ongoing,
                    result.Resolved,
                    result.BlindMetrics,
                    result.UnevaluableMetrics);

                // Named, not counted. "5 metrics returned nothing" tells an operator that the guard is partly
                // blind and nothing about which query to go and fix; the names are the whole actionable part,
                // and they are cheap because the window already knows.
                //
                // Only for metrics that HAVE a binding, though. An unbound one is silent by construction, was
                // named once at startup, and cannot change without a config edit — warning about it every cycle
                // taught operators here to skip the line that the bound-but-silent case shares.
                var silentButBound = 0;

                if (result.BlindMetrics > 0)
                {
                    for (var m = 0; m < (int)MetricIndex.Count; m++)
                    {
                        var metric = (MetricIndex)m;

                        if (window.PodsReporting(metric) > 0 || _unbound[m])
                        {
                            continue;
                        }

                        silentButBound++;
                        _blindMetric(_logger, metric.ToString(), null);
                    }

                    // The custom channels, which this loop walked past until 2026-08-10 (XC-9's sibling,
                    // XC-11). The built-in half was well built and its comment argues the case exactly —
                    // "the names are the whole actionable part" — and then the loop was keyed on MetricIndex,
                    // so a client's own channels were counted in `blind=N` and never named. That is the half
                    // of the system a client is most likely to extend, and on the lab it is already five
                    // channels.
                    //
                    // No unbound case to skip here: a custom channel exists BECAUSE somebody wrote a binding
                    // for it, so every silent one is actionable by definition. The built-ins need that check
                    // only because the enum carries members no deployment has bound.
                    var custom = _options.Guard.CustomMetrics;

                    for (var c = 0; c < custom.Count; c++)
                    {
                        var name = custom[c].Name;

                        if (window.PodsReporting(name) > 0)
                        {
                            continue;
                        }

                        silentButBound++;
                        _blindMetric(_logger, name, null);
                    }
                }

                // Fires on the actionable count, not the total. `blind=N` on the cycle line above still
                // carries every blind spot including the unbound ones — this warning is about the ones
                // somebody can do something about today.
                if (silentButBound > 0 || result.PartialMetrics > 0)
                {
                    _blind(
                        _logger, silentButBound, result.PartialMetrics,
                        window.Pods.Count, result.Incidents, null);
                }

                // Pods left out of this window because they stopped reporting before it ended. Almost always
                // a rollout, a scale-down or a delete — and in that case saying so once beats a finding
                // naming a replica the operator cannot find. But a pod whose scraping broke while it kept
                // serving looks identical from here and is urgent, so the names go out rather than a count.
                var stale = _source.StalePodsExcluded;

                if (stale.Count > 0)
                {
                    _logger.LogInformation(
                        CycleEvent,
                        "{Count} pod(s) stopped reporting before the end of this window and were left out of "
                        + "it: {Pods}. Expected during a rollout or a scale-down; if one of these is still "
                        + "serving traffic then its scraping is broken, which looks like health from here.",
                        stale.Count,
                        string.Join(", ", stale));
                }

                // A warning rather than the cycle line, and repeated every cycle it persists. The counter
                // `overfit_guard_state_failures_total` is the alertable form of this, but an operator
                // reading logs after a restart that reopened everything needs to find the cause here rather
                // than infer it from a series they were not watching at the time.
                if (_guard.StateError is { } stateError)
                {
                    _logger.LogWarning(
                        CycleEvent,
                        "durable state is not being written: {StateError}. Incidents and calibration will "
                        + "not survive the next restart.",
                        stateError);
                }

                ProposeFloors(window, now);

                return GuardCycleOutcome.Completed(result);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                // Shutting down. Not a failure, and not something to log as one.
                throw;
            }
            catch (Exception ex)
            {
                // Counted as well as logged. A loop that fails every cycle updates no incident counter, so
                // without its own series the only evidence is a log line nobody is watching — and no
                // incidents is exactly what a healthy cluster looks like.
                _guard.Telemetry.Failed();
                _cycleFailed(_logger, ex);

                return GuardCycleOutcome.Failed;
            }
        }
    }
}
