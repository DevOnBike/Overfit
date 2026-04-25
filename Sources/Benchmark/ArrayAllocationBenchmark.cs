// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;

namespace Benchmarks
{
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ArrayAllocationBenchmark
    {
        // Parametryzacja rozmiarów tablicy: 
        // 1024 (1 KB), 65 536 (64 KB), 1 048 576 (1 MB), 10 485 760 (10 MB)
        [Params(1024, 65536, 1_048_576, 10_485_760)]
        public int Size { get; set; }

        [Benchmark(Baseline = true)]
        public float[] StandardNew()
        {
            // Klasyczna alokacja: system daje pamięć i od razu wpisuje w nią zera.
            return new float[Size];
        }

        [Benchmark]
        public float[] AllocateUninitialized()
        {
            // Hack HPC: system daje "brudną" pamięć. Zero wpisywania.
            return GC.AllocateUninitializedArray<float>(Size);
        }

        [Benchmark]
        public void ArrayPool_RentAndClear()
        {
            // Wypożyczamy pamięć z puli i ręcznie czyścimy 
            // (Symulacja bezpiecznego reużywania buforów)
            var array = ArrayPool<float>.Shared.Rent(Size);
            Array.Clear(array, 0, Size);
            ArrayPool<float>.Shared.Return(array);
        }

        [Benchmark]
        public void ArrayPool_RentNoClear()
        {
            // Wypożyczamy pamięć z puli i oddajemy brudną.
            var array = ArrayPool<float>.Shared.Rent(Size);
            ArrayPool<float>.Shared.Return(array);
        }
    }
}