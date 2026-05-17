// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class ArrayAllocationBenchmark
    {
        // Array size parameterization:
        // 1024 (1 KB), 65 536 (64 KB), 1 048 576 (1 MB), 10 485 760 (10 MB)
        [Params(1024, 65536, 1_048_576, 10_485_760)]
        public int Size { get; set; }

        [Benchmark(Baseline = true)]
        public float[] StandardNew()
        {
            // Classic allocation: the system provides memory and immediately zeroes it.
            return new float[Size];
        }

        [Benchmark]
        public float[] AllocateUninitialized()
        {
            // HPC hack: the system returns "dirty" memory. No zeroing.
            return GC.AllocateUninitializedArray<float>(Size);
        }

        [Benchmark]
        public void ArrayPool_RentAndClear()
        {
            // Rent memory from the pool and clear it manually
            // (simulates safe buffer reuse)
            var array = ArrayPool<float>.Shared.Rent(Size);
            Array.Clear(array, 0, Size);
            ArrayPool<float>.Shared.Return(array);
        }

        [Benchmark]
        public void ArrayPool_RentNoClear()
        {
            // Rent memory from the pool and return it dirty.
            var array = ArrayPool<float>.Shared.Rent(Size);
            ArrayPool<float>.Shared.Return(array);
        }
    }
}