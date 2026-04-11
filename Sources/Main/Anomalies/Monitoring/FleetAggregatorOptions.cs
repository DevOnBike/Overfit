// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    public sealed class FleetAggregatorOptions
    {
        /// <summary>Must match AlignerOptions.WindowSize.</summary>
        public int WindowSize { get; init; } = 60;

        /// <summary>Must match AlignerOptions.MetricCount.</summary>
        public int MetricCount { get; init; } = 12;
    }
}