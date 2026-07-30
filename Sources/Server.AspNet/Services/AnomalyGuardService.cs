// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
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
        private const int CycleFailedEventId = 5005;
        private const int TopologyStaleEventId = 5006;

        private static readonly Action<ILogger, int, int, int, int, Exception?> _blind =
            LoggerMessage.Define<int, int, int, int>(
                LogLevel.Warning,
                new EventId(BlindEventId, "AnomalyGuardPartiallyBlind"),
                "Anomaly guard is partly blind: {BlindMetrics} metric(s) returned nothing for any pod and "
                + "{PartialMetrics} for only some, across {Pods} pods — an absence of incidents this cycle "
                + "({Incidents}) does not mean the cluster is healthy");

        // Its own event, because stale topology degrades grouping quietly rather than loudly: the guard keeps
        // producing incidents, they are just related to each other by out-of-date coordinates.
        private static readonly Action<ILogger, int, Exception?> _topologyStale =
            LoggerMessage.Define<int>(
                LogLevel.Warning,
                new EventId(TopologyStaleEventId, "AnomalyGuardTopologyStale"),
                "Pod topology could not be refreshed; grouping this cycle used the previous snapshot of "
                + "{Pods} pod(s). A pod created since then falls back to a name heuristic.");

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
        private int _knownPods;

        public AnomalyGuardService(
            AnomalyGuardServiceOptions options,
            PrometheusMetricWindowSource source,
            IIncidentSink sink,
            ILogger<AnomalyGuardService> logger,
            IRefreshablePodTopology? topology = null)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(source);
            ArgumentNullException.ThrowIfNull(sink);
            ArgumentNullException.ThrowIfNull(logger);

            _options = options;
            _source = source;
            _topology = topology;
            _logger = logger;

            // The guard reads topology through the interface, so handing it the same instance the loop
            // refreshes is what keeps the two in step — no snapshot is copied anywhere.
            _guard = new AnomalyGuard(
                options.Guard with { PodTopology = topology },
                sink,
                options.Tracking);
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
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

                if (result.BlindMetrics > 0 || result.PartialMetrics > 0)
                {
                    _blind(
                        _logger, result.BlindMetrics, result.PartialMetrics,
                        window.Pods.Count, result.Incidents, null);
                }
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
