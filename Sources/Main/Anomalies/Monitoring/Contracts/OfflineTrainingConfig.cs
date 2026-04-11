// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com


// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    ///     Immutable configuration for <see cref="OfflineTrainingJob" />.
    ///     Create with object initializer syntax:
    ///     <code>
    ///   var config = new OfflineTrainingConfig { Epochs = 100, LearningRate = 5e-4f };
    /// </code>
    /// </summary>
    public sealed record OfflineTrainingConfig
    {
        /// <summary>Number of full passes over the training data. Default: 50.</summary>
        public int Epochs { get; init; } = 50;

        /// <summary>Adam learning rate. Default: 0.001.</summary>
        public float LearningRate { get; init; } = 1e-3f;

        /// <summary>
        ///     Percentile used when calibrating the <see cref="ReconstructionScorer" /> threshold.
        ///     Default: 0.99 (p99 of training MSE values).
        /// </summary>
        public float CalibrationPercentile { get; init; } = 0.99f;

        /// <summary>
        ///     Whether to shuffle training samples before each epoch.
        ///     Disable for reproducible deterministic runs. Default: true.
        /// </summary>
        public bool ShuffleEachEpoch { get; init; } = true;

        /// <summary>
        ///     Optional random seed for sample shuffling.
        ///     null = non-deterministic. Set to a fixed value for reproducible training.
        /// </summary>
        public int? Seed { get; init; }
    }
}