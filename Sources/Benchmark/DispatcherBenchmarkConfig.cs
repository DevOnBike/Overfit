using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;

namespace Benchmarks
{
    /// <summary>
    /// Config for <see cref="OverfitParallelForBenchmark"/>. Mirrors the shared
    /// <c>BenchmarkConfig</c> but omits the <c>WithInvocationCount(1)</c> /
    /// <c>WithUnrollFactor(1)</c> pin so BDN auto-scales the invocation count —
    /// required to time a µs-scale dispatch call without noise.
    /// </summary>
    internal sealed class DispatcherBenchmarkConfig : ManualConfig
    {
        public DispatcherBenchmarkConfig()
        {
            AddJob(Job.Default
                .WithWarmupCount(5)
                .WithIterationCount(20));

            AddDiagnoser(MemoryDiagnoser.Default);
            AddColumn(RankColumn.Arabic);

            WithOrderer(new DefaultOrderer(SummaryOrderPolicy.FastestToSlowest));
        }
    }
}