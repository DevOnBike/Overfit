// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// Benchmarks the SGD optimizer path after moving the update rule through
    /// ElementwiseKernels.MultiplyAdd(...).
    ///
    /// This benchmark measures optimizer hot-path behavior only. Optimizer
    /// construction and parameter allocation are outside the measured region.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class SgdOptimizerBenchmark : IDisposable
    {
        private const int SmallParameterSize = 1024;
        private const int LargeParameterSize = 1_048_576;

        private const int SmallOperationsPerInvoke = 4_194_304;
        private const int LargeOperationsPerInvoke = 1024;

        private AutogradNode _smallParameter = null!;
        private AutogradNode _largeParameter = null!;

        private SGD _smallOptimizer = null!;
        private SGD _largeOptimizer = null!;

        [GlobalSetup]
        public void Setup()
        {
            _smallParameter = CreateParameter(
                SmallParameterSize,
                dataSeedOffset: 1,
                gradSeedOffset: 2);

            _largeParameter = CreateParameter(
                LargeParameterSize,
                dataSeedOffset: 3,
                gradSeedOffset: 4);

            _smallOptimizer = new SGD(
                new[] { _smallParameter },
                learningRate: 0.001f);

            _largeOptimizer = new SGD(
                new[] { _largeParameter },
                learningRate: 0.001f);

            for (var i = 0; i < 128; i++)
            {
                _smallOptimizer.Step();
                _largeOptimizer.Step();
            }
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float Sgd_Step_Small1024()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                _smallOptimizer.Step();
                checksum += _smallParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float Sgd_Step_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                _largeOptimizer.Step();
                checksum += _largeParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            Dispose();
        }

        public void Dispose()
        {
            _smallParameter?.Dispose();
            _largeParameter?.Dispose();
        }

        private static AutogradNode CreateParameter(
            int length,
            int dataSeedOffset,
            int gradSeedOffset)
        {
            var storage = new TensorStorage<float>(
                length,
                clearMemory: false);

            FillDeterministic(
                storage.AsSpan(),
                dataSeedOffset);

            var node = new AutogradNode(
                storage,
                TensorShape.Vector(length),
                requiresGrad: true);

            FillDeterministic(
                node.GradView.AsSpan(),
                gradSeedOffset);

            return node;
        }

        private static void FillDeterministic(
            Span<float> data,
            int seedOffset)
        {
            var seed = 0x12345678u + (uint)(seedOffset * 7919);

            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;

                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }
    }
}
