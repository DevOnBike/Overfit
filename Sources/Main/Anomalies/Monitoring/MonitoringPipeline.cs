// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    ///     Offline training pipeline for the anomaly detection autoencoder.
    ///     Processes raw Prometheus metric series from the Golden Window into
    ///     scaled tensors ready for LSTM autoencoder training.
    /// </summary>
    public sealed class MonitoringPipeline
    {
        private readonly FleetAggregator _aggregator;
        private readonly TimeSeriesAligner _aligner;
        private readonly WindowSanitizer _sanitizer;

        // ZASTĄPIONO: RobustScalingLayer na CompositeNormalizationLayer
        private readonly CompositeNormalizationLayer _baselineScaler;
        private readonly CompositeNormalizationLayer _deviationScaler;

        private bool _isTrainingPhase = true; // Tryb pracy: True = Golden Window (uczy się)

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

            // Inicjalizacja nowych warstw z dedykowanymi strategiami normalizacji
            _baselineScaler = new CompositeNormalizationLayer();
            _deviationScaler = new CompositeNormalizationLayer();
        }

        public ScaledResult Process(List<RawMetricSeries> allSeries, long globalStartMs, long scrapeTimestampMs)
        {
            var alignResult = _aligner.Align(allSeries, globalStartMs);

            _sanitizer.Sanitize(alignResult, scrapeTimestampMs);

            var aggResult = _aggregator.Aggregate(alignResult);

            // 1. Normalizujemy Medianę Floty (Baseline)
            _baselineScaler.ProcessInPlace(aggResult.FleetBaseline, aggResult.DcCount * aggResult.WindowSize, aggResult.MetricCount, _isTrainingPhase);

            // 2. Normalizujemy Odchylenia Podów od Mediany
            _deviationScaler.ProcessInPlace(aggResult.PodDeviations, aggResult.PodCount * aggResult.WindowSize, aggResult.MetricCount, _isTrainingPhase);

            var baselineTensor = new FastTensor<float>(aggResult.DcCount, aggResult.WindowSize, aggResult.MetricCount);

            aggResult.FleetBaseline.AsSpan().CopyTo(baselineTensor.AsSpan());

            var deviationsTensor = new FastTensor<float>(aggResult.PodCount, aggResult.WindowSize, aggResult.MetricCount);

            aggResult.PodDeviations.AsSpan().CopyTo(deviationsTensor.AsSpan());

            // Zwracamy wynik bezpiecznie zapakowany w Tensory
            return new ScaledResult
            {
                FleetBaseline = baselineTensor,
                PodDeviations = deviationsTensor,
                PodIndex = aggResult.PodIndex
            };
        }

        /// <summary>
        /// Wywołaj tę metodę, gdy faza przetwarzania Golden Window dobiegnie końca.
        /// Zamraża to wariancje, średnie i ekstrema do działania na produkcji (Inferencja).
        /// </summary>
        public void FinalizeTrainingPhase()
        {
            _baselineScaler.FreezeAll();
            _deviationScaler.FreezeAll();

            _isTrainingPhase = false;
        }

        public void ResetScalers()
        {
            _baselineScaler.ResetAll();
            _deviationScaler.ResetAll();

            _isTrainingPhase = true;
        }
    }
}