// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;
namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    /// Packs AggregationResult into FastTensors and applies RobustScalingLayer.
    ///
    /// FleetBaseline:  float[2, 60, 12]  →  FastTensor [2×60, 12]  →  scaled in place
    /// PodDeviations:  float[N, 60, 12]  →  FastTensor [N×60, 12]  →  scaled in place
    ///
    /// Each timestep of each pod/DC is treated as a separate sample row.
    /// The scaler normalises each of the 12 metric columns independently.
    /// </summary>
    public static class AggregationScaler
    {
        /// <summary>
        /// Scales FleetBaseline and PodDeviations in place using the provided scalers.
        /// Scalers must already be fitted (offline on Golden Window data).
        /// Returns FastTensors ready for the LSTM autoencoder.
        /// </summary>
        public static ScaledResult Scale(
            AggregationResult aggregation,
            RobustScalingLayer baselineScaler,
            RobustScalingLayer deviationScaler)
        {
            var baselineTensor = PackToTensor(aggregation.FleetBaseline, aggregation.DcCount * aggregation.WindowSize, aggregation.MetricCount);
            var deviationTensor = PackToTensor(aggregation.PodDeviations, aggregation.PodCount * aggregation.WindowSize, aggregation.MetricCount);

            baselineScaler.Process(WrapInContext(baselineTensor));
            deviationScaler.Process(WrapInContext(deviationTensor));

            return new ScaledResult
            {
                FleetBaseline = baselineTensor.Reshape(aggregation.DcCount, aggregation.WindowSize, aggregation.MetricCount),
                PodDeviations = deviationTensor.Reshape(aggregation.PodCount, aggregation.WindowSize, aggregation.MetricCount),
                PodIndex = aggregation.PodIndex
            };
        }

        private static FastTensor<float> PackToTensor(float[] data, int rows, int cols)
        {
            var tensor = new FastTensor<float>(false, rows, cols);
            var span = tensor.AsSpan();

            data.AsSpan().CopyTo(span);

            return tensor;
        }

        private static PipelineContext WrapInContext(FastTensor<float> tensor)
        {
            // Targets not used by RobustScalingLayer — pass empty tensor as placeholder
            var emptyTargets = new FastTensor<float>(false, 0);

            return new PipelineContext(tensor, emptyTargets);
        }
    }
}