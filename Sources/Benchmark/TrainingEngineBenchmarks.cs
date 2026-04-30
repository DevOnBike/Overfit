// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Training;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class TrainingEngineBenchmarks : IDisposable
    {
        private const int BatchSize = 64;
        private const int InputSize = 784;
        private const int HiddenSize = 128;
        private const int ClassCount = 10;

        // ~1.6 ms * 64 = ~102 ms per BDN iteration.
        private const int OperationsPerInvoke = 64;

        private float[] _input = null!;
        private float[] _target = null!;

        private Sequential _model = null!;
        private TrainingEngine _trainer = null!;

        private class Config : ManualConfig
        {
            public Config()
            {
                AddJob(Job.Default
                    .WithWarmupCount(5)
                    .WithIterationCount(20)
                    .WithInvocationCount(1)
                    .WithUnrollFactor(1));

                AddDiagnoser(MemoryDiagnoser.Default);
            }
        }

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            _input = new float[BatchSize * InputSize];
            _target = new float[BatchSize * ClassCount];

            FillDeterministic(_input);
            FillDeterministicOneHotTargets(_target, BatchSize, ClassCount);

            _model = new Sequential(
                new LinearLayer(InputSize, HiddenSize),
                new ReluActivation(),
                new LinearLayer(HiddenSize, ClassCount));

            var adam = new Adam(
                _model.Parameters(),
                learningRate: 0.001f);

            var optimizer = new DelegateTrainingOptimizer(
                zeroGrad: adam.ZeroGrad,
                step: adam.Step);

            var loss = new DelegateTrainingLoss(
                forward: (graph, prediction, target) =>
                    TensorMath.SoftmaxCrossEntropy(
                        graph,
                        prediction,
                        target),

                backward: (graph, lossNode) =>
                    graph.Backward(lossNode));

            _trainer = TrainingEngine.FromBackend(
                new SequentialTrainingBackend(
                    _model,
                    optimizer,
                    loss,
                    BatchSize,
                    InputSize,
                    ClassCount,
                    new TrainingEngineOptions
                    {
                        ResetGraphAfterStep = true,
                        ValidateFiniteInput = false,
                        ValidateFiniteTarget = false,
                        DisposeModelWithEngine = false
                    }));

            // Warmup outside measured region.
            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _trainer.TrainBatch(_input, _target);
            }
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float TrainingEngine_Mlp_TrainBatch()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                checksum += _trainer.TrainBatch(
                    _input,
                    _target).Loss;
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _trainer?.Dispose();
            _model?.Dispose();
        }

        public void Dispose()
        {
            Cleanup();
        }

        private static void FillDeterministic(
            float[] data)
        {
            var seed = 0x12345678u;

            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;

                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }

        private static void FillDeterministicOneHotTargets(
            float[] target,
            int batchSize,
            int classCount)
        {
            Array.Clear(target);

            for (var b = 0; b < batchSize; b++)
            {
                var cls = b % classCount;
                target[b * classCount + cls] = 1f;
            }
        }
    }
}