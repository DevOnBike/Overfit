// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Training.Contracts
{
    public sealed class TrainingEngineOptions
    {
        /// <summary>
        /// If true, input spans are checked for NaN/Infinity.
        /// Useful for debugging, not recommended for max-performance training.
        /// </summary>
        public bool ValidateFiniteInput { get; init; }

        /// <summary>
        /// If true, target spans are checked for NaN/Infinity.
        /// Useful for debugging, not recommended for max-performance training.
        /// </summary>
        public bool ValidateFiniteTarget { get; init; }

        /// <summary>
        /// If true, backend disposes the model when the engine is disposed.
        /// Default false, because model ownership usually belongs to caller.
        /// </summary>
        public bool DisposeModelWithEngine { get; init; }

        /// <summary>
        /// If true, graph.Reset() is called after every training step.
        /// Default true.
        /// </summary>
        public bool ResetGraphAfterStep { get; init; } = true;
    }
}
