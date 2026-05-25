// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Training
{
    /// <summary>Result returned after OfflineTrainingJob.RunAsync completes.</summary>
    public sealed class OfflineTrainingResult
    {
        public int SnapshotsLoaded { get; init; }
        public int SkippedCsvRows { get; init; }
        public float InitialLoss { get; init; }
        public float FinalValLoss { get; init; }
        public string CheckpointPath { get; init; } = string.Empty;
        public TimeSpan TrainingTime { get; init; }

        public override string ToString() =>
            $"Snapshots: {SnapshotsLoaded:N0} | ValLoss: {FinalValLoss:F4} | " +
            $"Time: {TrainingTime:mm\\:ss} | Checkpoint: {CheckpointPath}";
    }
}
