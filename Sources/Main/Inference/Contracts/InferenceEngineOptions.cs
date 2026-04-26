// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Inference.Contracts
{
    public sealed class InferenceEngineOptions
    {
        public const int DefaultMinimumWorkspaceElements = 64 * 1024;

        /// <summary>
        /// Explicit workspace size used by Sequential.PrepareInference(...).
        /// If null, the engine uses a conservative default.
        /// </summary>
        public int? MaxIntermediateElements { get; init; }

        /// <summary>
        /// Number of dry runs executed during engine initialization.
        /// Used to warm up JIT and layer caches.
        /// </summary>
        public int WarmupIterations { get; init; } = 4;

        /// <summary>
        /// If true, disposing the backend also disposes the wrapped Sequential model.
        /// Default is false to avoid surprising ownership transfer.
        /// </summary>
        public bool DisposeModelWithEngine { get; init; }

        /// <summary>
        /// If true, Run/Predict validates that the input does not contain NaN/Infinity.
        /// This is useful in debug scenarios but costs CPU.
        /// </summary>
        public bool ValidateFiniteInput { get; init; }

        internal int ResolveWorkspaceElements(
            int inputSize,
            int outputSize)
        {
            if (MaxIntermediateElements.HasValue)
            {
                if (MaxIntermediateElements.Value <= 0)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(MaxIntermediateElements),
                        "MaxIntermediateElements must be positive.");
                }

                return MaxIntermediateElements.Value;
            }

            var sizeBasedFallback = Math.Max(inputSize, outputSize) * 4;

            return Math.Max(
                DefaultMinimumWorkspaceElements,
                sizeBasedFallback);
        }
    }
}
