// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Training.Contracts;

namespace DevOnBike.Overfit.Training
{
    public interface ITrainingBackend : IDisposable
    {
        int BatchSize { get; }

        int InputSize { get; }

        int TargetSize { get; }

        TrainingStepResult TrainBatch(ReadOnlySpan<float> input, ReadOnlySpan<float> target);
    }
}
