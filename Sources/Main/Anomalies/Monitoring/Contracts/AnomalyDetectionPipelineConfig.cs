// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    /// Immutable configuration used by <see cref="AnomalyDetectionPipeline.Create"/>.
    /// </summary>
    public sealed record AnomalyDetectionPipelineConfig
    {
        /// <summary>
        /// Number of metric snapshots that form one feature window.
        /// Default: 6 (60 seconds at 10-second scrape interval).
        /// </summary>
        public int WindowSize { get; init; } = 6;

        /// <summary>
        /// Number of new samples required before the window slides forward.
        /// Default: 1 — a new window is produced on every sample once the buffer is full.
        /// </summary>
        public int StepSize { get; init; } = 1;

        /// <summary>
        /// Number of metric features per snapshot.
        /// Must match <see cref="AnomalyAutoencoder.InputSize"/> / <see cref="FeatureExtractor.StatsPerFeature"/>.
        /// Default: <see cref="MetricSnapshot.FeatureCount"/> (8).
        /// </summary>
        public int FeatureCount { get; init; } = MetricSnapshot.FeatureCount;
    }
}