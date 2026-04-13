// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    public sealed class MonitoringPipeline
    {
        private readonly FleetAggregator _aggregator;
        private readonly TimeSeriesAligner _aligner;
        private readonly WindowSanitizer _sanitizer;

        private readonly CompositeNormalizationLayer _baselineScaler;
        private readonly CompositeNormalizationLayer _deviationScaler;

        private bool _isTrainingPhase = true;

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

            _baselineScaler = new CompositeNormalizationLayer();
            _deviationScaler = new CompositeNormalizationLayer();
        }

        public ScaledResult Process(List<RawMetricSeries> allSeries, long globalStartMs, long scrapeTimestampMs)
        {
            var alignResult = _aligner.Align(allSeries, globalStartMs);

            _sanitizer.Sanitize(alignResult, scrapeTimestampMs);

            var aggResult = _aggregator.Aggregate(alignResult);

            _baselineScaler.ProcessInPlace(aggResult.FleetBaseline, aggResult.DcCount * aggResult.WindowSize, aggResult.MetricCount, _isTrainingPhase);
            _deviationScaler.ProcessInPlace(aggResult.PodDeviations, aggResult.PodCount * aggResult.WindowSize, aggResult.MetricCount, _isTrainingPhase);

            // Alokujemy nowe tensory jako magazyny (nie musimy ich czyścić zerami, bo zaraz nadpiszemy)
            var baselineTensor = new FastTensor<float>(aggResult.DcCount, aggResult.WindowSize, aggResult.MetricCount, clearMemory: false);
            aggResult.FleetBaseline.AsSpan().CopyTo(baselineTensor.GetView().AsSpan());

            var deviationsTensor = new FastTensor<float>(aggResult.PodCount, aggResult.WindowSize, aggResult.MetricCount, clearMemory: false);
            aggResult.PodDeviations.AsSpan().CopyTo(deviationsTensor.GetView().AsSpan());

            return new ScaledResult
            {
                FleetBaseline = baselineTensor,
                PodDeviations = deviationsTensor,
                PodIndex = aggResult.PodIndex
            };
        }

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