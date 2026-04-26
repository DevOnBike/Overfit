// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference.Contracts;

namespace DevOnBike.Overfit.Inference
{
    public sealed class SequentialInferenceBackend : IInferenceBackend
    {
        private readonly Sequential _model;
        private readonly bool _disposeModelWithEngine;
        private readonly float[] _warmupInput;
        private readonly float[] _warmupOutput;
        private bool _disposed;

        public SequentialInferenceBackend(
            Sequential model,
            int inputSize,
            int outputSize,
            InferenceEngineOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            _model = model;

            var resolvedOptions = options ?? new InferenceEngineOptions();
            _disposeModelWithEngine = resolvedOptions.DisposeModelWithEngine;

            InputSize = inputSize;
            OutputSize = outputSize;

            var workspaceElements = resolvedOptions.ResolveWorkspaceElements(
                inputSize,
                outputSize);

            _model.Eval();
            _model.PrepareInference(workspaceElements);

            _warmupInput = new float[inputSize];
            _warmupOutput = new float[outputSize];
        }

        public int InputSize { get; }

        public int OutputSize { get; }

        public void Run(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            ThrowIfDisposed();

            _model.ForwardInference(
                input,
                output);
        }

        public void Warmup(int iterations)
        {
            ThrowIfDisposed();

            if (iterations <= 0)
            {
                return;
            }

            for (var i = 0; i < iterations; i++)
            {
                _model.ForwardInference(
                    _warmupInput,
                    _warmupOutput);
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            if (_disposeModelWithEngine)
            {
                _model.Dispose();
            }

            _disposed = true;
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
