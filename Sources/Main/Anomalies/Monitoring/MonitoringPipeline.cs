// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Data.Prepare;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Offline training pipeline for the anomaly detection autoencoder.
    /// Processes raw Prometheus metric series from the Golden Window into
    /// scaled tensors ready for LSTM autoencoder training.
    ///
    /// Flow per batch:
    ///   List&lt;RawMetricSeries&gt;
    ///     → TimeSeriesAligner   (align to shared time grid)
    ///     → WindowSanitizer     (remove invalid pods, correct values)
    ///     → FleetAggregator     (fleet medians + pod deviations)
    ///     → AggregationScaler   (RobustScaler fit+transform)
    ///     → ScaledResult        (FastTensor ready for training)
    /// </summary>
    public sealed class MonitoringPipeline
    {
        private readonly TimeSeriesAligner _aligner;
        private readonly WindowSanitizer _sanitizer;
        private readonly FleetAggregator _aggregator;
        private readonly RobustScalingLayer _baselineScaler;
        private readonly RobustScalingLayer _deviationScaler;

        public MonitoringPipeline(MonitoringPipelineOptions options)
        {
            _aligner = new TimeSeriesAligner(new AlignerOptions
            {
                WindowSize = options.WindowSize,
                StepSeconds = options.StepSeconds,
                MetricCount = options.MetricCount,
                MaxGapSteps = options.MaxGapSteps
            });

            _sanitizer = new WindowSanitizer(new SanitizerOptions
            {
                WarmupDuration = options.WarmupDuration,
                MaxNanRatio = options.MaxNanRatio
            });

            _aggregator = new FleetAggregator(new FleetAggregatorOptions
            {
                WindowSize = options.WindowSize,
                MetricCount = options.MetricCount
            });

            _baselineScaler = new RobustScalingLayer();
            _deviationScaler = new RobustScalingLayer();
        }

        /// <summary>
        /// Processes one batch of raw metric series.
        /// On first call the scalers fit themselves on the incoming data.
        /// On subsequent calls they apply the fitted parameters (Transform only).
        /// </summary>
        public ScaledResult Process(List<RawMetricSeries> allSeries, long globalStartMs, long scrapeTimestampMs)
        {
            var alignResult = _aligner.Align(allSeries, globalStartMs);

            _sanitizer.Sanitize(alignResult, scrapeTimestampMs);

            var aggResult = _aggregator.Aggregate(alignResult);

            return AggregationScaler.Scale(aggResult, _baselineScaler, _deviationScaler);
        }

        // ---------------------------------------------------------------------------
        // Scaler persistence — call after Golden Window processing is complete
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Resets both scalers — forces re-fit on next Process call.
        /// Use when starting a new Golden Window collection.
        /// </summary>
        public void ResetScalers()
        {
            _baselineScaler.Reset();
            _deviationScaler.Reset();
        }
    }
}