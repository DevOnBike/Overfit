// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com


// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Anomalies.Alerting.Contracts
{
    /// <summary>Immutable snapshot of a single anomaly detection event.</summary>
    public sealed record AlertEvent
    {
        /// <summary>Name of the K8s pod that triggered the alert.</summary>
        public required string PodName { get; init; }

        /// <summary>Normalised anomaly score ∈ [0, 1] from <see cref="ReconstructionScorer"/>.</summary>
        public float AnomalyScore { get; init; }

        /// <summary>Raw MSE(input, reconstruction) before threshold normalisation.</summary>
        public float ReconstructionMse { get; init; }

        /// <summary>UTC timestamp when the alert was detected.</summary>
        public DateTime DetectedAt { get; init; }

        /// <summary>Warning or Critical based on score vs configured thresholds.</summary>
        public AlertSeverity Severity { get; init; }
    }
}