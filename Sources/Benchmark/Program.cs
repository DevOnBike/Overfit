// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Running;

namespace Benchmarks
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            // BenchmarkRunner.Run<SingleInferenceBenchmark>();
            BenchmarkRunner.Run<InferenceBenchmark>();
            // BenchmarkRunner.Run<MultiLayerInferenceBenchmark>();
            // BenchmarkRunner.Run<TailLatencyBenchmark>();
            // BenchmarkRunner.Run<ColdStartBenchmark>();
            // BenchmarkRunner.Run<ConcurrentInferenceBenchmark>();
            // BenchmarkRunner.Run<ThroughputBenchmark>();
            // BenchmarkRunner.Run<ScalingBenchmark>();
        }
    }
}