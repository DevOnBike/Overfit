// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring.Contracts
{
    /// <summary>
    /// Current metric values for a single monitored pod.
    /// Returned by <see cref="AnomalyMetricsCollector.GetSnapshot"/>.
    /// </summary>
    public sealed record PodMetricsSnapshot
    {
        public required string PodName { get; init; }
        public float AnomalyScore { get; init; }
        public float ReconstructionMse { get; init; }
        public long WindowsProcessed { get; init; }
        public long AlertsFired { get; init; }
        public DateTime LastUpdated { get; init; }
    }
}