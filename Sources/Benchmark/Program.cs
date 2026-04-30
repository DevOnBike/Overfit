// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Running;
using DevOnBike.Overfit.Licensing;

namespace Benchmarks
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            OverfitLicense.SuppressNotice = true;
            OverfitLicense.MessageSink = _ => { };

            // BenchmarkRunner.Run<SingleInferenceBenchmark>();
            // BenchmarkRunner.Run<InferenceBenchmark>();
            // BenchmarkRunner.Run<MultiLayerInferenceBenchmark>();
            // BenchmarkRunner.Run<MLNetSingleInferenceBenchmark>();
            // BenchmarkRunner.Run<TailLatencyBenchmark>();
            // BenchmarkRunner.Run<ColdStartBenchmark>();
            // BenchmarkRunner.Run<ConcurrentInferenceBenchmark>();
            // BenchmarkRunner.Run<ThroughputBenchmark>();
            // BenchmarkRunner.Run<ScalingBenchmark>();

            // BenchmarkRunner.Run<OverfitKernelBenchmarks>();
            // BenchmarkRunner.Run<ThreadScalingBenchmarks>();
            // BenchmarkRunner.Run<MathNetInferenceBattleBenchmark>();
            // BenchmarkRunner.Run<BatchScalingBenchmark>();
            //BenchmarkRunner.Run<GenerationalGeneticAlgorithmBenchmarks>();
            // BenchmarkRunner.Run<EvolutionaryAlgorithmBenchmarks>();

            //BenchmarkRunner.Run<InferenceZeroAllocBenchmarks>();
            //BenchmarkRunner.Run<OnnxCnnInferenceBenchmarks>();
            //BenchmarkRunner.Run<OnnxMlpInferenceBenchmarks>();

            //BenchmarkRunner.Run<BatchScalingBenchmark>();
            BenchmarkRunner.Run<OnnxGraphBenchmark>();
        }
    }
}