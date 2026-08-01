// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// The loop: read a window, evaluate it, report what changed, wait, repeat.
    ///
    /// <para>Everything it does lives elsewhere — <see cref="PrometheusMetricWindowSource"/> fetches,
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

        private static readonly Action<ILogger, Exception?> _cycleFailed =
            LoggerMessage.Define(
                LogLevel.Error,
                new EventId(CycleFailedEventId, "AnomalyGuardCycleFailed"),
                "Anomaly guard cycle failed; skipping it and continuing.");

        private readonly AnomalyGuardServiceOptions _options;
        private readonly PrometheusMetricWindowSource _source;
        private readonly IRefreshablePodTopology? _topology;
        private readonly AnomalyGuard _guard;
        private readonly ILogger<AnomalyGuardService> _logger;
        private readonly FloorCalibrator? _calibrator;
        private int _knownPods;
        private DateTimeOffset _nextProposal;

        /// <param name="store">
        /// Optional durable state. <b>Supply one in any deployment that can be restarted</b>, which is all of
        /// them: without it every incident that was running is reopened after a rollout or a crash, and the
        /// operator is paged again for problems they were already told about — the tracker's whole
        /// contribution undone by the guard's own restart. This was missing here while
        /// <see cref="AnomalyGuard"/> had supported it all along, so the durable path existed and nothing
        /// deployable reached it.
        /// </param>
        public AnomalyGuardService(
            AnomalyGuardServiceOptions options,
            PrometheusMetricWindowSource source,
            IIncidentSink sink,
            ILogger<AnomalyGuardService> logger,
            IRefreshablePodTopology? topology = null,
            IIncidentStore? store = null)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(source);
            ArgumentNullException.ThrowIfNull(sink);
            ArgumentNullException.ThrowIfNull(logger);

            _options = options;
            _source = source;
            _topology = topology;
            _logger = logger;

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
                store);
        }

        /// <summary>Incidents adopted from durable state when this instance started. Zero without a store.</summary>
        public int RestoredIncidents => _guard.RestoredIncidents;

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
                await RunOneCycleAsync(stoppingToken).ConfigureAwait(false);
            }
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
            if (_topology is null)
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
            if (_calibrator is null)
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

                if (!proposal.IsUsable || proposal.ProposedMinAbsoluteGap <= 0.0)
                {
                    continue;
                }

                var metric = (MetricIndex)m;

                // Both gates, because they are read by different families and only one of them was being
                // reported. The peer gate governs how far apart two replicas may sit; the trend gate governs
                // how far one may move across a window. On the lab the dominant false-positive source is the
                // TREND family, so proposing only the peer floor addressed the smaller half of the problem —
                // and did it silently, which is worse than not addressing it.
                Propose(
                    metric, "peer gap",
                    AnomalyGuardOptions.FloorFor(_options.Guard.MinAbsoluteGap, metric),
                    proposal.PeerGapMax, proposal.ProposedMinAbsoluteGap, proposal);

                Propose(
                    metric, "trend change across a window",
                    AnomalyGuardOptions.FloorFor(_options.Guard.MinAbsoluteTrendChange, metric),
                    proposal.TrendChangeMax, proposal.ProposedMinAbsoluteTrendChange, proposal);
            }
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

        private async Task RunOneCycleAsync(CancellationToken ct)
        {
            try
            {
                await RefreshTopologyAsync(ct).ConfigureAwait(false);

                var now = DateTimeOffset.UtcNow;
                var end = now - _options.EndOffset;
                var window = await _source.ReadAsync(end, _options.Window, ct).ConfigureAwait(false);

                if (window is null)
                {
                    // No pod returned anything. Not an empty cluster — a cluster this source cannot see.
                    _blind(_logger, (int)MetricIndex.Count, 0, 0, 0, null);

                    return;
                }

                var result = _guard.RunCycle(window, now);

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
            if (result.BlindMetrics > 0)
            {
                for (var m = 0; m < (int)MetricIndex.Count; m++)
                {
                    var metric = (MetricIndex)m;

                    if (window.PodsReporting(metric) == 0)
                    {
                        _blindMetric(_logger, metric.ToString(), null);
                    }
                }
            }

                if (result.BlindMetrics > 0 || result.PartialMetrics > 0)
                {
                    _blind(
                        _logger, result.BlindMetrics, result.PartialMetrics,
                        window.Pods.Count, result.Incidents, null);
                }

                ProposeFloors(window, now);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                // Shutting down. Not a failure, and not something to log as one.
                throw;
            }
            catch (Exception ex)
            {
                _cycleFailed(_logger, ex);
            }
        }
    }
}
