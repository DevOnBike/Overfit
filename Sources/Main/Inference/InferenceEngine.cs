// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference.Contracts;

namespace DevOnBike.Overfit.Inference
{
    public sealed class InferenceEngine : IDisposable
    {
        private readonly IInferenceBackend _backend;
        private readonly InferenceEngineOptions _options;
        private readonly float[] _singleOutput;
        private bool _disposed;

        private InferenceEngine(
            IInferenceBackend backend,
            InferenceEngineOptions options)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _singleOutput = new float[backend.OutputSize];

            if (options.WarmupIterations > 0)
            {
                backend.Warmup(options.WarmupIterations);
            }
        }

        public int InputSize => _backend.InputSize;

        public int OutputSize => _backend.OutputSize;

        /// <summary>
        /// Creates an inference engine from a Sequential model.
        /// Shape is explicit by design because current Sequential does not expose
        /// reliable model-level input/output metadata.
        /// </summary>
        public static InferenceEngine FromSequential(
            Sequential model,
            int inputSize,
            int outputSize,
            InferenceEngineOptions? options = null)
        {
            var resolvedOptions = options ?? new InferenceEngineOptions();

            var backend = new SequentialInferenceBackend(
                model,
                inputSize,
                outputSize,
                resolvedOptions);

            return new InferenceEngine(
                backend,
                resolvedOptions);
        }

        /// <summary>
        /// Creates an inference engine from a custom backend.
        /// Extension point for ONNX/custom/GPU/compiled backends.
        /// </summary>
        public static InferenceEngine FromBackend(
            IInferenceBackend backend,
            InferenceEngineOptions? options = null)
        {
            return new InferenceEngine(
                backend,
                options ?? new InferenceEngineOptions());
        }

        /// <summary>
        /// Runs one prediction using an internal preallocated output buffer.
        /// The returned span is overwritten on the next Predict call.
        /// </summary>
        public ReadOnlySpan<float> Predict(ReadOnlySpan<float> input)
        {
            ThrowIfDisposed();

            if (input.Length != InputSize)
            {
                throw new ArgumentException(
                    $"Predict expects exactly one sample: input length {InputSize}, got {input.Length}.",
                    nameof(input));
            }

            if (_options.ValidateFiniteInput)
            {
                ValidateFinite(input);
            }

            _backend.Run(
                input,
                _singleOutput);

            return _singleOutput;
        }

        /// <summary>
        /// Runs inference into a caller-provided output buffer.
        /// Supports single-sample and batched input.
        /// </summary>
        public void Run(
            ReadOnlySpan<float> input,
            Span<float> output)
        {
            ThrowIfDisposed();

            if (input.Length % InputSize != 0)
            {
                throw new ArgumentException(
                    $"Input length must be divisible by InputSize={InputSize}. Got {input.Length}.",
                    nameof(input));
            }

            var batchSize = input.Length / InputSize;
            var requiredOutputLength = batchSize * OutputSize;

            if (output.Length < requiredOutputLength)
            {
                throw new ArgumentException(
                    $"Output span is too small. Required {requiredOutputLength}, got {output.Length}.",
                    nameof(output));
            }

            if (_options.ValidateFiniteInput)
            {
                ValidateFinite(input);
            }

            _backend.Run(
                input,
                output);
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

        private static void ValidateFinite(ReadOnlySpan<float> values)
        {
            for (var i = 0; i < values.Length; i++)
            {
                if (!float.IsFinite(values[i]))
                {
                    throw new ArgumentException(
                        $"Input contains non-finite value at index {i}: {values[i]}.");
                }
            }
        }
    }
}