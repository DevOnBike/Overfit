// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Orchestrates the full anomaly detection pipeline for a single K8s pod:
    ///
    ///   IMetricSource → SlidingWindowBuffer → FeatureExtractor
    ///       → AnomalyAutoencoder → ReconstructionScorer
    ///       → AlertEngine → AnomalyMetricsCollector
    ///
    /// Ownership:
    ///   The pipeline does NOT own the components passed to its constructor —
    ///   callers retain responsibility for disposing them after <see cref="RunAsync"/>
    ///   returns. Only the <see cref="SlidingWindowBuffer"/> created by
    ///   <see cref="Create"/> is owned and disposed by the pipeline.
    ///
    /// Usage (DI constructor):
    /// <code>
    ///   var pipeline = new AnomalyDetectionPipeline(source, buffer, autoencoder, scorer);
    ///   await pipeline.RunAsync(stoppingToken);
    /// </code>
    ///
    /// Usage (factory):
    /// <code>
    ///   await using var pipeline = AnomalyDetectionPipeline.Create(
    ///       source, config, autoencoder, scorer, alertEngine, metrics);
    ///   await pipeline.RunAsync(stoppingToken);
    /// </code>
    /// </summary>
    public sealed class AnomalyDetectionPipeline : IAsyncDisposable
    {
        // ---- required components ----
        private readonly IMetricSource _source;
        private readonly SlidingWindowBuffer _buffer;
        private readonly AnomalyAutoencoder _autoencoder;
        private readonly ReconstructionScorer _scorer;

        // ---- optional components ----
        private readonly AlertEngine? _alertEngine;
        private readonly AnomalyMetricsCollector? _metrics;

        // ---- per-call pre-allocated buffers (zero allocation on hot path) ----
        private readonly float[] _windowScratch;        // [WindowSize × FeatureCount]
        private readonly float[] _statsScratch;         // [FeatureCount × StatsPerFeature]
        private readonly float[] _reconstructionScratch; // [InputSize]

        // ---- lifecycle ----
        private readonly bool _ownsBuffer;

        // ---- diagnostics ----
        private long _windowsProcessed;
        private long _alertsFired;

        /// <summary>Total feature windows processed since <see cref="RunAsync"/> started.</summary>
        public long WindowsProcessed => Volatile.Read(ref _windowsProcessed);

        /// <summary>Total alerts fired since <see cref="RunAsync"/> started.</summary>
        public long AlertsFired => Volatile.Read(ref _alertsFired);

        // -------------------------------------------------------------------------
        // Constructors
        // -------------------------------------------------------------------------

        /// <summary>
        /// DI constructor — all components are pre-built and caller-owned.
        /// The pipeline does not dispose any of them.
        /// </summary>
        public AnomalyDetectionPipeline(
            IMetricSource source,
            SlidingWindowBuffer buffer,
            AnomalyAutoencoder autoencoder,
            ReconstructionScorer scorer,
            AlertEngine? alertEngine = null,
            AnomalyMetricsCollector? metrics = null)
            : this(source, buffer, autoencoder, scorer, alertEngine, metrics, ownsBuffer: false)
        { }

        private AnomalyDetectionPipeline(
            IMetricSource source,
            SlidingWindowBuffer buffer,
            AnomalyAutoencoder autoencoder,
            ReconstructionScorer scorer,
            AlertEngine? alertEngine,
            AnomalyMetricsCollector? metrics,
            bool ownsBuffer)
        {
            ArgumentNullException.ThrowIfNull(source);
            ArgumentNullException.ThrowIfNull(buffer);
            ArgumentNullException.ThrowIfNull(autoencoder);
            ArgumentNullException.ThrowIfNull(scorer);

            _source = source;
            _buffer = buffer;
            _autoencoder = autoencoder;
            _scorer = scorer;
            _alertEngine = alertEngine;
            _metrics = metrics;
            _ownsBuffer = ownsBuffer;

            // Pre-allocate inference buffers once — reused across every RunAsync iteration
            _windowScratch = new float[buffer.WindowFloats];
            _statsScratch = new float[FeatureExtractor.OutputSize(buffer.FeatureCount)];
            _reconstructionScratch = new float[autoencoder.InputSize];
        }

        // -------------------------------------------------------------------------
        // Factory
        // -------------------------------------------------------------------------

        /// <summary>
        /// Creates a pipeline with a freshly built <see cref="SlidingWindowBuffer"/>.
        /// The returned pipeline owns the buffer and disposes it on <see cref="DisposeAsync"/>.
        /// All other components remain caller-owned.
        /// </summary>
        public static AnomalyDetectionPipeline Create(
            IMetricSource source,
            AnomalyDetectionPipelineConfig config,
            AnomalyAutoencoder autoencoder,
            ReconstructionScorer scorer,
            AlertEngine? alertEngine = null,
            AnomalyMetricsCollector? metrics = null)
        {
            ArgumentNullException.ThrowIfNull(config);

            var buffer = new SlidingWindowBuffer(
                config.WindowSize,
                config.StepSize,
                config.FeatureCount);

            return new AnomalyDetectionPipeline(
                source, buffer, autoencoder, scorer,
                alertEngine, metrics, ownsBuffer: true);
        }

        // -------------------------------------------------------------------------
        // RunAsync — main loop
        // -------------------------------------------------------------------------

        /// <summary>
        /// Runs the inference loop until <paramref name="ct"/> is cancelled.
        ///
        /// Per iteration:
        ///   1. Read next <see cref="MetricSnapshot"/> from the source (blocks until available).
        ///   2. Push snapshot into the sliding window buffer.
        ///   3. When a full window is ready: extract features, reconstruct, score.
        ///   4. Optionally fire alerts and record Prometheus metrics.
        ///
        /// Returns normally on cancellation — does not throw <see cref="OperationCanceledException"/>.
        /// </summary>
        /// <param name="onInference">
        ///   Optional callback invoked for every completed window.
        ///   Useful for logging, dashboards, or integration tests.
        /// </param>
        public async Task RunAsync(
            CancellationToken ct = default,
            Action<InferenceResult>? onInference = null)
        {
            while (!ct.IsCancellationRequested)
            {
                MetricSnapshot snapshot;

                try
                {
                    snapshot = await _source.ReadAsync(ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                _buffer.Add(in snapshot);

                if (!_buffer.TryGetWindow(_windowScratch, out var windowEnd))
                {
                    continue;
                }

                // Feature extraction: window → per-feature [mean, std, p95, delta]
                FeatureExtractor.Extract(
                    _windowScratch,
                    _buffer.WindowSize,
                    _buffer.FeatureCount,
                    _statsScratch);

                // Reconstruction through the autoencoder
                _autoencoder.Reconstruct(_statsScratch, _reconstructionScratch);

                // Scoring: compute MSE once, derive normalised score from threshold
                var mse = ReconstructionScorer.ComputeMse(_statsScratch, _reconstructionScratch);
                var score = _scorer.ComputeScore(mse);

                Interlocked.Increment(ref _windowsProcessed);

                // Optional: expose metrics to Prometheus before alerting
                _metrics?.RecordInference(snapshot.PodName, score, mse);

                // Optional: alert if score exceeds threshold and pod is not in cooldown
                if (_alertEngine?.TryAlert(snapshot.PodName, score, mse) == true)
                {
                    Interlocked.Increment(ref _alertsFired);
                    _metrics?.RecordAlert(snapshot.PodName);
                }

                // Optional: caller-provided callback (logging, integration tests)
                onInference?.Invoke(new InferenceResult
                {
                    PodName = snapshot.PodName,
                    AnomalyScore = score,
                    ReconstructionMse = mse,
                    WindowEnd = windowEnd
                });
            }
        }

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        /// <summary>
        /// Disposes the <see cref="SlidingWindowBuffer"/> if created by <see cref="Create"/>.
        /// All other components remain the caller's responsibility.
        /// </summary>
        public ValueTask DisposeAsync()
        {
            if (_ownsBuffer)
            {
                _buffer.Dispose();
            }

            return ValueTask.CompletedTask;
        }
    }
}