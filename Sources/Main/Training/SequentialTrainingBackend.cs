// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Training.Contracts;

namespace DevOnBike.Overfit.Training
{
    public sealed class SequentialTrainingBackend : ITrainingBackend
    {
        private readonly Sequential _model;
        private readonly ITrainingOptimizer _optimizer;
        private readonly ITrainingLoss _loss;
        private readonly TrainingEngineOptions _options;

        private readonly ComputationGraph _graph;

        private readonly TensorStorage<float> _inputStorage;
        private readonly TensorStorage<float> _targetStorage;

        private readonly AutogradNode _inputNode;
        private readonly AutogradNode _targetNode;

        private bool _disposed;

        public SequentialTrainingBackend(
            Sequential model,
            ITrainingOptimizer optimizer,
            ITrainingLoss loss,
            int batchSize,
            int inputSize,
            int targetSize,
            TrainingEngineOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(optimizer);
            ArgumentNullException.ThrowIfNull(loss);

            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetSize);

            _model = model;
            _optimizer = optimizer;
            _loss = loss;
            _options = options ?? new TrainingEngineOptions();

            BatchSize = batchSize;
            InputSize = inputSize;
            TargetSize = targetSize;

            _graph = new ComputationGraph();

            _inputStorage = new TensorStorage<float>(
                batchSize * inputSize,
                clearMemory: false);

            _targetStorage = new TensorStorage<float>(
                batchSize * targetSize,
                clearMemory: false);

            _inputNode = new AutogradNode(
                _inputStorage,
                new TensorShape(batchSize, inputSize),
                requiresGrad: false);

            _targetNode = new AutogradNode(
                _targetStorage,
                new TensorShape(batchSize, targetSize),
                requiresGrad: false);

            _model.Train();
        }

        public int BatchSize { get; }

        public int InputSize { get; }

        public int TargetSize { get; }

        public TrainingStepResult TrainBatch(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> target)
        {
            ThrowIfDisposed();

            if (input.Length != BatchSize * InputSize)
            {
                throw new ArgumentException(
                    $"Expected input length {BatchSize * InputSize}, got {input.Length}.",
                    nameof(input));
            }

            if (target.Length != BatchSize * TargetSize)
            {
                throw new ArgumentException(
                    $"Expected target length {BatchSize * TargetSize}, got {target.Length}.",
                    nameof(target));
            }

            input.CopyTo(_inputStorage.AsSpan());
            target.CopyTo(_targetStorage.AsSpan());

            _optimizer.ZeroGrad();

            var prediction = _model.Forward(
                _graph,
                _inputNode);

            var lossNode = _loss.Forward(
                _graph,
                prediction,
                _targetNode);

            var lossValue = _loss.ReadScalar(lossNode);

            _loss.Backward(
                _graph,
                lossNode);

            _optimizer.Step();

            if (_options.ResetGraphAfterStep)
            {
                _graph.Reset();
            }

            return new TrainingStepResult(
                lossValue,
                BatchSize,
                input.Length,
                target.Length);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _graph.Dispose();

            _inputNode.Dispose();
            _targetNode.Dispose();

            _inputStorage.Dispose();
            _targetStorage.Dispose();

            if (_options.DisposeModelWithEngine)
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
