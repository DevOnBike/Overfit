// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Concurrent;
using System.Threading.Channels;
using DevOnBike.Overfit.Anomalies.Alerting.Abstractions;
using DevOnBike.Overfit.Anomalies.Alerting.Contracts;

namespace DevOnBike.Overfit.Anomalies.Alerting
{
    /// <summary>
    ///     Evaluates anomaly scores against configured thresholds and dispatches
    ///     alert events to registered <see cref="IAlertSink" /> implementations.
    ///     Hot-path contract:
    ///     <see cref="TryAlert" /> is synchronous and non-blocking.
    ///     Sink calls are dispatched via an internal <see cref="Channel{T}" /> and
    ///     executed by a background consumer task — the scoring loop is never delayed
    ///     by slow sinks, network timeouts, or sink failures.
    ///     Cooldown:
    ///     Alerts for the same pod are suppressed until <see cref="AlertEngineConfig.CooldownDuration" />
    ///     has elapsed since the last fired alert. This prevents alert storms
    ///     during sustained anomaly periods.
    ///     Usage:
    ///     <code>
    ///   await using var engine = new AlertEngine(config, teamsWebhookSink, pagerDutySink);
    ///   // ...in scoring loop:
    ///   var score = scorer.Score(features, reconstruction);
    ///   engine.TryAlert(podName, score, mse);
    /// </code>
    /// </summary>
    public sealed class AlertEngine : IAsyncDisposable
    {
        private readonly AlertEngineConfig _config;
        private readonly Task _consumer;
        private readonly CancellationTokenSource _cts = new();

        // Per-pod last-alert timestamp stored as UTC ticks for lock-free compare
        private readonly ConcurrentDictionary<string, long> _lastAlertTicks = new();

        // Bounded channel isolates the hot-path from slow sinks.
        // DropOldest: if sinks fall behind, recent alerts take priority over old ones.
        private readonly Channel<AlertEvent> _queue;
        private readonly IReadOnlyList<IAlertSink> _sinks;

        // Diagnostics
        private long _alertsFired;
        private long _alertsSuppressed;

        /// <param name="config">Engine configuration. Pass null for defaults.</param>
        /// <param name="sinks">One or more alert sinks. Must not be empty.</param>
        /// <exception cref="ArgumentException">When sinks is empty.</exception>
        public AlertEngine(AlertEngineConfig config, params IAlertSink[] sinks)
        {
            ArgumentNullException.ThrowIfNull(sinks);

            if (sinks.Length == 0)
            {
                throw new ArgumentException("At least one alert sink is required.", nameof(sinks));
            }

            _config = config ?? new AlertEngineConfig();
            _sinks = sinks;

            if (_config.AlertThreshold <= 0f || _config.AlertThreshold > 1f)
            {
                throw new ArgumentException($"AlertThreshold must be in (0, 1], got {_config.AlertThreshold}.", nameof(config));
            }

            if (_config.CriticalThreshold < _config.AlertThreshold)
            {
                throw new ArgumentException(
                $"CriticalThreshold ({_config.CriticalThreshold}) must be >= " +
                $"AlertThreshold ({_config.AlertThreshold}).",
                nameof(config));
            }

            _queue = Channel.CreateBounded<AlertEvent>(new BoundedChannelOptions(_config.DispatchQueueCapacity)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = false,
                AllowSynchronousContinuations = false
            });

            _consumer = ConsumeAsync();
        }

        /// <summary>Total alerts dispatched to the queue since construction.</summary>
        public long AlertsFired => Volatile.Read(ref _alertsFired);

        /// <summary>Alerts suppressed due to being below threshold or within cooldown.</summary>
        public long AlertsSuppressed => Volatile.Read(ref _alertsSuppressed);

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        /// <summary>
        ///     Completes the dispatch queue, waits for all pending alerts to be
        ///     delivered, then disposes background resources.
        /// </summary>
        public async ValueTask DisposeAsync()
        {
            // Signal no more alerts will be written, then wait for the consumer
            // to drain all pending events before returning.
            _queue.Writer.TryComplete();
            await _consumer.ConfigureAwait(false);
            _cts.Dispose();
        }

        // -------------------------------------------------------------------------
        // Hot-path
        // -------------------------------------------------------------------------

        /// <summary>
        ///     Evaluates the anomaly score and enqueues an alert if conditions are met.
        ///     Returns true if an alert was enqueued.
        ///     Returns false if score is below threshold or pod is within cooldown.
        ///     This method is synchronous and non-blocking — safe to call on the
        ///     inference thread at scraping frequency (≥1/10 s).
        /// </summary>
        /// <param name="podName">K8s pod identifier.</param>
        /// <param name="anomalyScore">Score ∈ [0, 1] from <see cref="ReconstructionScorer" />.</param>
        /// <param name="reconstructionMse">Raw MSE for diagnostic purposes.</param>
        public bool TryAlert(string podName, float anomalyScore, float reconstructionMse)
        {
            ArgumentException.ThrowIfNullOrEmpty(podName);

            if (anomalyScore < _config.AlertThreshold)
            {
                Interlocked.Increment(ref _alertsSuppressed);
                return false;
            }

            if (IsInCooldown(podName))
            {
                Interlocked.Increment(ref _alertsSuppressed);
                return false;
            }

            var alert = new AlertEvent
            {
                PodName = podName,
                AnomalyScore = anomalyScore,
                ReconstructionMse = reconstructionMse,
                DetectedAt = DateTime.UtcNow,
                Severity = anomalyScore >= _config.CriticalThreshold ? AlertSeverity.Critical : AlertSeverity.Warning
            };

            // Non-blocking write — TryWrite never waits
            _queue.Writer.TryWrite(alert);

            RecordAlertTime(podName);
            Interlocked.Increment(ref _alertsFired);

            return true;
        }

        // -------------------------------------------------------------------------
        // Background consumer
        // -------------------------------------------------------------------------

        private async Task ConsumeAsync()
        {
            // No cancellation token — drain the queue gracefully until writer is completed.
            // ReadAllAsync() completes naturally when TryComplete() is called and the queue is empty.
            await foreach (var alert in _queue.Reader.ReadAllAsync().ConfigureAwait(false))
            {
                await DispatchToSinksAsync(alert).ConfigureAwait(false);
            }
        }

        private async Task DispatchToSinksAsync(AlertEvent alert)
        {
            foreach (var sink in _sinks)
            {
                try
                {
                    await sink.SendAsync(alert, _cts.Token).ConfigureAwait(false);
                }
                catch
                {
                    // Sink failures are silently swallowed — a bad sink must not
                    // prevent delivery to other sinks or block the consumer loop.
                    // Implementors should log inside their IAlertSink.SendAsync.
                }
            }
        }

        // -------------------------------------------------------------------------
        // Cooldown helpers
        // -------------------------------------------------------------------------

        private bool IsInCooldown(string podName)
        {
            if (!_lastAlertTicks.TryGetValue(podName, out var lastTicks))
            {
                return false;
            }

            var elapsed = TimeSpan.FromTicks(DateTime.UtcNow.Ticks - lastTicks);
            return elapsed < _config.CooldownDuration;
        }

        private void RecordAlertTime(string podName)
        {
            _lastAlertTicks[podName] = DateTime.UtcNow.Ticks;
        }
    }

}