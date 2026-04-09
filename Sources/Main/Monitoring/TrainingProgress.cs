// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>Per-epoch snapshot reported via <see cref="IProgress{T}"/>.</summary>
    public sealed record TrainingProgress
    {
        /// <summary>Current epoch number (1-based).</summary>
        public int Epoch { get; init; }

        /// <summary>Total number of epochs configured.</summary>
        public int TotalEpochs { get; init; }

        /// <summary>Average MSE loss across all samples in this epoch.</summary>
        public float EpochLoss { get; init; }

        /// <summary>Completion percentage in [0, 100].</summary>
        public float ProgressPct => (float)Epoch / TotalEpochs * 100f;
    }
}