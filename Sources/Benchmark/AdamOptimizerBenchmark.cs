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
    /// Baseline benchmark for Adam before any ElementwiseKernels refactor.
    ///
    /// Keep this benchmark stable. It is the performance guard for future Adam cleanup.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class AdamOptimizerBenchmark
    {
        private const int SmallSize = 1024;
        private const int LargeSize = 1_048_576;

        private const int SmallOperationsPerInvoke = 524_288;
        private const int LargeOperationsPerInvoke = 1024;

        private AutogradNode _smallParameter = null!;
        private AutogradNode _largeParameter = null!;

        private Adam _smallAdamW = null!;
        private Adam _largeAdamW = null!;

        private Adam _smallCoupledAdam = null!;
        private Adam _largeCoupledAdam = null!;

        [GlobalSetup]
        public void Setup()
        {
            _smallParameter = CreateParameter(
                SmallSize,
                seedOffset: 1);

            _largeParameter = CreateParameter(
                LargeSize,
                seedOffset: 2);

            var smallCoupledParameter = CreateParameter(
                SmallSize,
                seedOffset: 3);

            var largeCoupledParameter = CreateParameter(
                LargeSize,
                seedOffset: 4);

            _smallAdamW = CreateAdam(
                _smallParameter,
                useAdamW: true);

            _largeAdamW = CreateAdam(
                _largeParameter,
                useAdamW: true);

            _smallCoupledAdam = CreateAdam(
                smallCoupledParameter,
                useAdamW: false);

            _largeCoupledAdam = CreateAdam(
                largeCoupledParameter,
                useAdamW: false);

            _smallAdamW.Step();
            _largeAdamW.Step();
            _smallCoupledAdam.Step();
            _largeCoupledAdam.Step();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _smallAdamW.Dispose();
            _largeAdamW.Dispose();
            _smallCoupledAdam.Dispose();
            _largeCoupledAdam.Dispose();

            _smallParameter.Dispose();
            _largeParameter.Dispose();

            // The coupled Adam optimizers own optimizer state only.
            // Their parameter nodes are intentionally kept private to the optimizer setup.
            // They will be disposed through process teardown in this benchmark.
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float AdamW_Step_Small1024()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                _smallAdamW.Step();
                checksum += _smallParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float AdamW_Step_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                _largeAdamW.Step();
                checksum += _largeParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float CoupledAdam_Step_Small1024()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                _smallCoupledAdam.Step();
                checksum += _smallParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float CoupledAdam_Step_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                _largeCoupledAdam.Step();
                checksum += _largeParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        private static Adam CreateAdam(
            AutogradNode parameter,
            bool useAdamW)
        {
            return new Adam(
                new[]
                {
                    parameter
                },
                learningRate: 0.001f)
            {
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                Epsilon = 1e-8f,
                WeightDecay = 0.0001f,
                UseAdamW = useAdamW
            };
        }

        private static AutogradNode CreateParameter(
            int length,
            int seedOffset)
        {
            var storage = new TensorStorage<float>(
                length,
                clearMemory: false);

            var data = storage.AsSpan();

            FillDeterministic(
                data,
                seedOffset);

            var node = new AutogradNode(
                storage,
                TensorShape.Vector(length),
                requiresGrad: true);

            FillDeterministic(
                node.GradView.AsSpan(),
                seedOffset + 100);

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