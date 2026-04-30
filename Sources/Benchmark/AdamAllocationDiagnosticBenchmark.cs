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
    [Config(typeof(BenchmarkConfig))]
    public class AdamAllocationDiagnosticBenchmark
    {
        private const int SmallSize = 128;
        private const int MediumSize = 1024;
        private const int LargeSize = 1_048_576;

        private const int SmallOperationsPerInvoke = 524_288;
        private const int MediumOperationsPerInvoke = 524_288;
        private const int LargeOperationsPerInvoke = 1024;
        private const int PowOperationsPerInvoke = 8_388_608;

        private AutogradNode _smallParameter = null!;
        private AutogradNode _mediumParameter = null!;
        private AutogradNode _largeParameter = null!;

        private FastTensor<float> _smallM = null!;
        private FastTensor<float> _smallV = null!;
        private FastTensor<float> _mediumM = null!;
        private FastTensor<float> _mediumV = null!;
        private FastTensor<float> _largeM = null!;
        private FastTensor<float> _largeV = null!;

        private Adam _smallAdam = null!;
        private Adam _mediumAdam = null!;
        private Adam _largeAdam = null!;

        private float _beta1;
        private float _beta2;
        private int _step;

        [GlobalSetup]
        public void Setup()
        {
            _beta1 = 0.9f;
            _beta2 = 0.999f;
            _step = 1;

            _smallParameter = CreateParameter(SmallSize, seedOffset: 1);
            _mediumParameter = CreateParameter(MediumSize, seedOffset: 2);
            _largeParameter = CreateParameter(LargeSize, seedOffset: 3);

            _smallM = CreateFastTensor(SmallSize, seedOffset: 11);
            _smallV = CreateFastTensor(SmallSize, seedOffset: 12);
            _mediumM = CreateFastTensor(MediumSize, seedOffset: 13);
            _mediumV = CreateFastTensor(MediumSize, seedOffset: 14);
            _largeM = CreateFastTensor(LargeSize, seedOffset: 15);
            _largeV = CreateFastTensor(LargeSize, seedOffset: 16);

            _smallAdam = CreateAdam(_smallParameter);
            _mediumAdam = CreateAdam(_mediumParameter);
            _largeAdam = CreateAdam(_largeParameter);

            _smallAdam.Step();
            _mediumAdam.Step();
            _largeAdam.Step();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _smallAdam.Dispose();
            _mediumAdam.Dispose();
            _largeAdam.Dispose();

            _smallParameter.Dispose();
            _mediumParameter.Dispose();
            _largeParameter.Dispose();

            _smallM.Dispose();
            _smallV.Dispose();
            _mediumM.Dispose();
            _mediumV.Dispose();
            _largeM.Dispose();
            _largeV.Dispose();
        }

        [Benchmark(OperationsPerInvoke = PowOperationsPerInvoke)]
        public float MathFPow_BiasCorrectionOnly()
        {
            var checksum = 0f;

            for (var i = 0; i < PowOperationsPerInvoke; i++)
            {
                var step = _step + i;

                var bc1 = 1f - MathF.Pow(_beta1, step);
                var bc2 = 1f - MathF.Pow(_beta2, step);

                checksum += bc1 + bc2;
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float DataAndGradSpanAccessOnly_Small128()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                var data = _smallParameter.DataView.AsReadOnlySpan();
                var grad = _smallParameter.GradView.AsReadOnlySpan();

                checksum += data[0] + grad[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MediumOperationsPerInvoke)]
        public float DataAndGradSpanAccessOnly_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < MediumOperationsPerInvoke; i++)
            {
                var data = _mediumParameter.DataView.AsReadOnlySpan();
                var grad = _mediumParameter.GradView.AsReadOnlySpan();

                checksum += data[0] + grad[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float DataAndGradSpanAccessOnly_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                var data = _largeParameter.DataView.AsReadOnlySpan();
                var grad = _largeParameter.GradView.AsReadOnlySpan();

                checksum += data[0] + grad[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float FastTensorStateSpanAccessOnly_Small128()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                var m = _smallM.AsSpan();
                var v = _smallV.AsSpan();

                checksum += m[0] + v[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MediumOperationsPerInvoke)]
        public float FastTensorStateSpanAccessOnly_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < MediumOperationsPerInvoke; i++)
            {
                var m = _mediumM.AsSpan();
                var v = _mediumV.AsSpan();

                checksum += m[0] + v[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float FastTensorStateSpanAccessOnly_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                var m = _largeM.AsSpan();
                var v = _largeV.AsSpan();

                checksum += m[0] + v[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float AdamStep_Small128()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                _smallAdam.Step();
                checksum += _smallParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MediumOperationsPerInvoke)]
        public float AdamStep_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < MediumOperationsPerInvoke; i++)
            {
                _mediumAdam.Step();
                checksum += _mediumParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float AdamStep_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                _largeAdam.Step();
                checksum += _largeParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float AdamStepWithResetTime_Small128()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                _smallAdam.ResetTime();
                _smallAdam.Step();

                checksum += _smallParameter.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        private static Adam CreateAdam(
            AutogradNode parameter)
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
                UseAdamW = true
            };
        }

        private static AutogradNode CreateParameter(
            int length,
            int seedOffset)
        {
            var storage = new TensorStorage<float>(
                length,
                clearMemory: false);

            FillDeterministic(
                storage.AsSpan(),
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

        private static FastTensor<float> CreateFastTensor(
            int length,
            int seedOffset)
        {
            var tensor = new FastTensor<float>(
                length,
                clearMemory: false);

            FillDeterministic(
                tensor.AsSpan(),
                seedOffset);

            return tensor;
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