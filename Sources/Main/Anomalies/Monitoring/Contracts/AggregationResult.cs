// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    ///     Aggregates sanitized pod windows into two datasets ready for the autoencoder:
    ///     1. FleetBaseline  — per-DC median across all pods, shape [dcCount, WindowSize, MetricCount]
    ///     Trains the autoencoder to recognise healthy fleet behaviour.
    ///     2. PodDeviations  — each pod minus its DC median, shape [podCount, WindowSize, MetricCount]
    ///     Trains the autoencoder to detect pods that diverge from the fleet.
    /// </summary>
    public sealed class AggregationResult
    {
        /// <summary>
        ///     Row-major [dcCount, WindowSize, MetricCount].
        ///     FleetBaseline[dc, t, m] = median of metric m at timestep t across all pods in DC dc.
        ///     Index 0 = DataCenter.West, 1 = DataCenter.East.
        /// </summary>
        public required float[] FleetBaseline { get; init; }

        /// <summary>
        ///     Row-major [podCount, WindowSize, MetricCount].
        ///     PodDeviations[pod, t, m] = pod_value - fleet_median for that DC.
        ///     Same pod ordering as PodIndex.
        /// </summary>
        public required float[] PodDeviations { get; init; }

        /// <summary>Pod identity — PodIndex[i] corresponds to PodDeviations row i.</summary>
        public required List<PodKey> PodIndex { get; init; }

        public int DcCount { get; init; }
        public int PodCount { get; init; }
        public int WindowSize { get; init; }
        public int MetricCount { get; init; }
    }

}