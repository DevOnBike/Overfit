// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Training;

namespace DevOnBike.Overfit.Tests
{
    public sealed class TrainingEngineTests
    {
        [Fact]
        public void TrainingEngine_TrainBatch_DelegatesToBackend()
        {
            var backend = new FakeTrainingBackend(
                batchSize: 2,
                inputSize: 3,
                targetSize: 1,
                loss: 0.125f);

            using var engine = TrainingEngine.FromBackend(backend);

            var input = new float[6];
            var target = new float[2];

            var result = engine.TrainBatch(
                input,
                target);

            Assert.Equal(1, backend.TrainBatchCalls);
            Assert.Equal(0.125f, result.Loss);
            Assert.Equal(2, result.BatchSize);
            Assert.Equal(6, result.InputLength);
            Assert.Equal(2, result.TargetLength);
        }

        [Fact]
        public void TrainingEngine_TrainBatch_RejectsWrongInputLength()
        {
            using var engine = TrainingEngine.FromBackend(
                new FakeTrainingBackend(
                    batchSize: 2,
                    inputSize: 3,
                    targetSize: 1,
                    loss: 1f));

            var input = new float[5];
            var target = new float[2];

            Assert.Throws<ArgumentException>(() =>
                engine.TrainBatch(
                    input,
                    target));
        }

        [Fact]
        public void TrainingEngine_TrainBatch_RejectsWrongTargetLength()
        {
            using var engine = TrainingEngine.FromBackend(
                new FakeTrainingBackend(
                    batchSize: 2,
                    inputSize: 3,
                    targetSize: 1,
                    loss: 1f));

            var input = new float[6];
            var target = new float[3];

            Assert.Throws<ArgumentException>(() =>
                engine.TrainBatch(
                    input,
                    target));
        }

        [Fact]
        public void TrainingEngine_TrainBatch_CanValidateFiniteInput()
        {
            using var engine = TrainingEngine.FromBackend(
                new FakeTrainingBackend(
                    batchSize: 1,
                    inputSize: 3,
                    targetSize: 1,
                    loss: 1f),
                new TrainingEngineOptions
                {
                    ValidateFiniteInput = true
                });

            var input = new[] { 1f, float.NaN, 3f };
            var target = new float[1];

            Assert.Throws<ArgumentException>(() =>
                engine.TrainBatch(
                    input,
                    target));
        }

        [Fact]
        public void DelegateTrainingOptimizer_CallsDelegates()
        {
            var zeroGradCalls = 0;
            var stepCalls = 0;

            var optimizer = new DelegateTrainingOptimizer(
                zeroGrad: () => zeroGradCalls++,
                step: () => stepCalls++);

            optimizer.ZeroGrad();
            optimizer.Step();

            Assert.Equal(1, zeroGradCalls);
            Assert.Equal(1, stepCalls);
        }

        private sealed class FakeTrainingBackend : ITrainingBackend
        {
            private readonly float _loss;
            private bool _disposed;

            public FakeTrainingBackend(
                int batchSize,
                int inputSize,
                int targetSize,
                float loss)
            {
                BatchSize = batchSize;
                InputSize = inputSize;
                TargetSize = targetSize;
                _loss = loss;
            }

            public int BatchSize { get; }

            public int InputSize { get; }

            public int TargetSize { get; }

            public int TrainBatchCalls { get; private set; }

            public TrainingStepResult TrainBatch(
                ReadOnlySpan<float> input,
                ReadOnlySpan<float> target)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                TrainBatchCalls++;

                return new TrainingStepResult(
                    _loss,
                    BatchSize,
                    input.Length,
                    target.Length);
            }

            public void Dispose()
            {
                _disposed = true;
            }
        }
    }
}
