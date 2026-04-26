// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Training
{
    public readonly struct TrainingStepResult
    {
        public TrainingStepResult(
            float loss,
            int batchSize,
            int inputLength,
            int targetLength)
        {
            Loss = loss;
            BatchSize = batchSize;
            InputLength = inputLength;
            TargetLength = targetLength;
        }

        public float Loss { get; }

        public int BatchSize { get; }

        public int InputLength { get; }

        public int TargetLength { get; }

        public override string ToString()
        {
            return $"Loss={Loss:0.000000}, BatchSize={BatchSize}";
        }
    }
}
