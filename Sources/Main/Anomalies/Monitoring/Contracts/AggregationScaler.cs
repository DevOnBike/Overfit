// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public static class AggregationScaler
    {
        public static ScaledResult Scale(
            AggregationResult aggregation,
            RobustScalingLayer baselineScaler,
            RobustScalingLayer deviationScaler)
        {
            using var baselineTensor = PackToTensor(aggregation.FleetBaseline, aggregation.DcCount * aggregation.WindowSize, aggregation.MetricCount);
            using var deviationTensor = PackToTensor(aggregation.PodDeviations, aggregation.PodCount * aggregation.WindowSize, aggregation.MetricCount);

            baselineScaler.Process(WrapInContext(baselineTensor));
            deviationScaler.Process(WrapInContext(deviationTensor));

            var reshapedBaseline = new FastTensor<float>(aggregation.DcCount, aggregation.WindowSize, aggregation.MetricCount, clearMemory: false);
            baselineTensor.GetView().AsReadOnlySpan().CopyTo(reshapedBaseline.GetView().AsSpan());

            var reshapedDeviations = new FastTensor<float>(aggregation.PodCount, aggregation.WindowSize, aggregation.MetricCount, clearMemory: false);
            deviationTensor.GetView().AsReadOnlySpan().CopyTo(reshapedDeviations.GetView().AsSpan());

            return new ScaledResult
            {
                FleetBaseline = reshapedBaseline,
                PodDeviations = reshapedDeviations,
                PodIndex = aggregation.PodIndex
            };
        }

        private static FastTensor<float> PackToTensor(float[] data, int rows, int cols)
        {
            var tensor = new FastTensor<float>(rows, cols, clearMemory: false);
            var span = tensor.GetView().AsSpan();

            data.AsSpan().CopyTo(span);

            return tensor;
        }

        private static PipelineContext WrapInContext(FastTensor<float> tensor)
        {
            // UWAGA ARCHITEKTONICZNA: Jeśli ten tensor o rozmiarze 0 nie jest nigdzie 
            // zwalniany wewnątrz PipelineContext, to zostawia po sobie mikroskopijny 
            // ślad dla Garbage Collectora. Jeśli to obciąża GC, warto rozważyć współdzielony 
            // statyczny pusty tensor, np. public static readonly FastTensor<float> Empty = ...
            var emptyTargets = new FastTensor<float>(0, clearMemory: false);

            return new PipelineContext(tensor, emptyTargets);
        }
    }
}