// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Kernels;

namespace Benchmarks
{
    /// <summary>
    /// Benchmarks for internal span-only elementwise kernels.
    ///
    /// These benchmarks are intentionally separated by operation size because
    /// very small spans need many operations per invocation to avoid
    /// BenchmarkDotNet MinIterationTime warnings.
    ///
    /// The goal is not to prove public API performance. The goal is to verify
    /// that the internal zero-allocation kernels are stable and suitable for
    /// future optimizer/loss/autograd cleanup work.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class ElementwiseKernelsBenchmark
    {
        private const int SmallLength = 16;
        private const int MediumLength = 1024;
        private const int LargeLength = 1_048_576;

        private const int SmallOperationsPerInvoke = 32_768_000;
        private const int MediumOperationsPerInvoke = 8_388_608;
        private const int ReluMediumOperationsPerInvoke = 524_288;
        private const int LargeOperationsPerInvoke = 1024;

        private float[] _xSmall = null!;
        private float[] _ySmall = null!;
        private float[] _zSmall = null!;
        private float[] _destinationSmall = null!;

        private float[] _xMedium = null!;
        private float[] _yMedium = null!;
        private float[] _zMedium = null!;
        private float[] _destinationMedium = null!;

        private float[] _xLarge = null!;
        private float[] _yLarge = null!;
        private float[] _zLarge = null!;
        private float[] _destinationLarge = null!;

        [GlobalSetup]
        public void Setup()
        {
            _xSmall = new float[SmallLength];
            _ySmall = new float[SmallLength];
            _zSmall = new float[SmallLength];
            _destinationSmall = new float[SmallLength];

            _xMedium = new float[MediumLength];
            _yMedium = new float[MediumLength];
            _zMedium = new float[MediumLength];
            _destinationMedium = new float[MediumLength];

            _xLarge = new float[LargeLength];
            _yLarge = new float[LargeLength];
            _zLarge = new float[LargeLength];
            _destinationLarge = new float[LargeLength];

            FillDeterministic(_xSmall, seedOffset: 1);
            FillDeterministic(_ySmall, seedOffset: 2);
            FillDeterministic(_zSmall, seedOffset: 3);

            FillDeterministic(_xMedium, seedOffset: 4);
            FillDeterministic(_yMedium, seedOffset: 5);
            FillDeterministic(_zMedium, seedOffset: 6);

            FillDeterministic(_xLarge, seedOffset: 7);
            FillDeterministic(_yLarge, seedOffset: 8);
            FillDeterministic(_zLarge, seedOffset: 9);

            ElementwiseKernels.Add(
                _xSmall,
                _ySmall,
                _destinationSmall);

            ElementwiseKernels.MultiplyAdd(
                _xSmall,
                0.125f,
                _zSmall,
                _destinationSmall);

            ElementwiseKernels.Add(
                _xMedium,
                _yMedium,
                _destinationMedium);

            ElementwiseKernels.MultiplyAdd(
                _xMedium,
                0.125f,
                _zMedium,
                _destinationMedium);

            ElementwiseKernels.ReLU(
                _xMedium,
                _destinationMedium);

            ElementwiseKernels.Add(
                _xLarge,
                _yLarge,
                _destinationLarge);

            ElementwiseKernels.MultiplyAdd(
                _xLarge,
                0.125f,
                _zLarge,
                _destinationLarge);
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float Add_Small16()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                ElementwiseKernels.Add(
                    _xSmall,
                    _ySmall,
                    _destinationSmall);

                checksum += _destinationSmall[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = SmallOperationsPerInvoke)]
        public float MultiplyAdd_Small16()
        {
            var checksum = 0f;

            for (var i = 0; i < SmallOperationsPerInvoke; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    _xSmall,
                    0.125f,
                    _zSmall,
                    _destinationSmall);

                checksum += _destinationSmall[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MediumOperationsPerInvoke)]
        public float Add_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < MediumOperationsPerInvoke; i++)
            {
                ElementwiseKernels.Add(
                    _xMedium,
                    _yMedium,
                    _destinationMedium);

                checksum += _destinationMedium[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MediumOperationsPerInvoke)]
        public float MultiplyAdd_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < MediumOperationsPerInvoke; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    _xMedium,
                    0.125f,
                    _zMedium,
                    _destinationMedium);

                checksum += _destinationMedium[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MediumOperationsPerInvoke)]
        public float Dot_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < MediumOperationsPerInvoke; i++)
            {
                checksum += ElementwiseKernels.Dot(
                    _xMedium,
                    _yMedium);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = MediumOperationsPerInvoke)]
        public float SumOfSquares_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < MediumOperationsPerInvoke; i++)
            {
                checksum += ElementwiseKernels.SumOfSquares(
                    _xMedium);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = ReluMediumOperationsPerInvoke)]
        public float ReLU_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < ReluMediumOperationsPerInvoke; i++)
            {
                ElementwiseKernels.ReLU(
                    _xMedium,
                    _destinationMedium);

                checksum += _destinationMedium[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float Add_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                ElementwiseKernels.Add(
                    _xLarge,
                    _yLarge,
                    _destinationLarge);

                checksum += _destinationLarge[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LargeOperationsPerInvoke)]
        public float MultiplyAdd_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < LargeOperationsPerInvoke; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    _xLarge,
                    0.125f,
                    _zLarge,
                    _destinationLarge);

                checksum += _destinationLarge[0];
            }

            return checksum;
        }

        private static void FillDeterministic(
            float[] data,
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