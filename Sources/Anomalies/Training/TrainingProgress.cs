// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Training
{
    /// <summary>Progress callback payload reported during <see cref="OfflineTrainingJob"/>.</summary>
    public sealed class TrainingProgress
    {
        public string Phase { get; init; } = string.Empty;
        public int Step
        {
            get; init;
        }
        public int TotalSteps
        {
            get; init;
        }
        public float TrainLoss
        {
            get; init;
        }
        public float ValLoss
        {
            get; init;
        }
        public TimeSpan Elapsed
        {
            get; init;
        }

        public override string ToString() =>
            $"[{Phase}] {Step}/{TotalSteps} train={TrainLoss:F4} val={ValLoss:F4} {Elapsed:mm\\:ss}";
    }
}
