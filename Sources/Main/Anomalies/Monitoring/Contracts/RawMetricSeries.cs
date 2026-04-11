// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public sealed class RawMetricSeries
    {
        public PodKey Pod { get; init; }

        public byte MetricTypeId { get; init; }

        public List<RawSample> Samples { get; } = [];

        public int Length => Samples.Count;
    }
}