// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Training
{
    public sealed class TrainingEngineOptions
    {
        public bool ValidateFiniteInput { get; init; }

        public bool ValidateFiniteTarget { get; init; }

        public bool DisposeModelWithEngine { get; init; }

        public bool ResetGraphAfterStep { get; init; } = true;
    }
}
