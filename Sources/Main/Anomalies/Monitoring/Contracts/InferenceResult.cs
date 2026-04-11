// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    ///     Result of a single inference step — emitted for every completed window.
    /// </summary>
    public sealed record InferenceResult
    {
        /// <summary>Name of the K8s pod that produced this result.</summary>
        public required string PodName { get; init; }

        /// <summary>Normalised anomaly score ∈ [0, 1].</summary>
        public float AnomalyScore { get; init; }

        /// <summary>Raw MSE(features, reconstruction) before threshold normalisation.</summary>
        public float ReconstructionMse { get; init; }

        /// <summary>UTC timestamp of the last sample in the window.</summary>
        public DateTime WindowEnd { get; init; }
    }
}