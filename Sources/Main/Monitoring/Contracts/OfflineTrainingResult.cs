// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring.Contracts
{
    /// <summary>Outcome returned by <see cref="OfflineTrainingJob.Run"/>.</summary>
    public sealed record OfflineTrainingResult
    {
        /// <summary>Average MSE loss per epoch, length == <see cref="OfflineTrainingConfig.Epochs"/>.</summary>
        public float[] EpochLosses { get; init; } = [];

        /// <summary>Calibrated anomaly threshold written to the scorer after training.</summary>
        public float FinalThreshold { get; init; }

        /// <summary>Wall-clock duration of the entire Run call.</summary>
        public TimeSpan Duration { get; init; }

        /// <summary>Loss at the first epoch — useful for sanity checks.</summary>
        public float InitialLoss => EpochLosses.Length > 0 ? EpochLosses[0] : 0f;

        /// <summary>Loss at the last epoch.</summary>
        public float FinalLoss => EpochLosses.Length > 0 ? EpochLosses[^1] : 0f;

        /// <summary>
        /// Fractional loss reduction from epoch 1 to last epoch.
        /// Positive = model improved. 0 if InitialLoss was zero.
        /// </summary>
        public float LossReduction => InitialLoss > 0f ? (InitialLoss - FinalLoss) / InitialLoss : 0f;
    }
}