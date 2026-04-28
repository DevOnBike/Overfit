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
    /// Experimental benchmark for Overfit's span-only element-wise kernels.
    ///
    /// This benchmark is intended to validate the small-span scalar path and
    /// larger-span TensorPrimitives path without touching model/training behavior.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class ElementwiseKernelsBenchmark
    {
        private const int OperationsPerInvoke = 32_768;

        private float[] _smallA = null!;
        private float[] _smallB = null!;
        private float[] _smallDestination = null!;

        private float[] _mediumA = null!;
        private float[] _mediumB = null!;
        private float[] _mediumDestination = null!;

        private float[] _largeA = null!;
        private float[] _largeB = null!;
        private float[] _largeDestination = null!;

        [GlobalSetup]
        public void Setup()
        {
            _smallA = new float[16];
            _smallB = new float[16];
            _smallDestination = new float[16];

            _mediumA = new float[1024];
            _mediumB = new float[1024];
            _mediumDestination = new float[1024];

            _largeA = new float[1 << 20];
            _largeB = new float[1 << 20];
            _largeDestination = new float[1 << 20];

            FillDeterministic(_smallA, 1);
            FillDeterministic(_smallB, 2);
            FillDeterministic(_mediumA, 3);
            FillDeterministic(_mediumB, 4);
            FillDeterministic(_largeA, 5);
            FillDeterministic(_largeB, 6);
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Add_Small16()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                ElementwiseKernels.Add(
                    _smallA,
                    _smallB,
                    _smallDestination);

                checksum += _smallDestination[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Add_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                ElementwiseKernels.Add(
                    _mediumA,
                    _mediumB,
                    _mediumDestination);

                checksum += _mediumDestination[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = 128)]
        public float Add_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < 128; i++)
            {
                ElementwiseKernels.Add(
                    _largeA,
                    _largeB,
                    _largeDestination);

                checksum += _largeDestination[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float MultiplyAdd_Small16()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    _smallA,
                    0.5f,
                    _smallB,
                    _smallDestination);

                checksum += _smallDestination[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float MultiplyAdd_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    _mediumA,
                    0.5f,
                    _mediumB,
                    _mediumDestination);

                checksum += _mediumDestination[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = 128)]
        public float MultiplyAdd_Large1M()
        {
            var checksum = 0f;

            for (var i = 0; i < 128; i++)
            {
                ElementwiseKernels.MultiplyAdd(
                    _largeA,
                    0.5f,
                    _largeB,
                    _largeDestination);

                checksum += _largeDestination[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float ReLU_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                ElementwiseKernels.ReLU(
                    _mediumA,
                    _mediumDestination);

                checksum += _mediumDestination[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Dot_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                checksum += ElementwiseKernels.Dot(
                    _mediumA,
                    _mediumB);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float SumOfSquares_Medium1024()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                checksum += ElementwiseKernels.SumOfSquares(_mediumA);
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
