// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Training
{
    public sealed class TrainingEngine : IDisposable
    {
        private readonly ITrainingBackend _backend;
        private readonly TrainingEngineOptions _options;
        private bool _disposed;

        private TrainingEngine(
            ITrainingBackend backend,
            TrainingEngineOptions options)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _options = options ?? throw new ArgumentNullException(nameof(options));
        }

        public int BatchSize => _backend.BatchSize;

        public int InputSize => _backend.InputSize;

        public int TargetSize => _backend.TargetSize;

        public static TrainingEngine FromBackend(
            ITrainingBackend backend,
            TrainingEngineOptions? options = null)
        {
            return new TrainingEngine(
                backend,
                options ?? new TrainingEngineOptions());
        }

        public TrainingStepResult TrainBatch(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> target)
        {
            ThrowIfDisposed();

            var expectedInputLength = BatchSize * InputSize;
            var expectedTargetLength = BatchSize * TargetSize;

            if (input.Length != expectedInputLength)
            {
                throw new ArgumentException(
                    $"Expected input length {expectedInputLength}, got {input.Length}.",
                    nameof(input));
            }

            if (target.Length != expectedTargetLength)
            {
                throw new ArgumentException(
                    $"Expected target length {expectedTargetLength}, got {target.Length}.",
                    nameof(target));
            }

            if (_options.ValidateFiniteInput)
            {
                ValidateFinite(
                    input,
                    nameof(input));
            }

            if (_options.ValidateFiniteTarget)
            {
                ValidateFinite(
                    target,
                    nameof(target));
            }

            return _backend.TrainBatch(
                input,
                target);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _backend.Dispose();
            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }

        private static void ValidateFinite(
            ReadOnlySpan<float> values,
            string argumentName)
        {
            for (var i = 0; i < values.Length; i++)
            {
                if (!float.IsFinite(values[i]))
                {
                    throw new ArgumentException(
                        $"{argumentName} contains non-finite value at index {i}: {values[i]}",
                        argumentName);
                }
            }
        }
    }
}
