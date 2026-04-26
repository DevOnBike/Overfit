using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;

namespace Benchmarks.Helpers
{
    internal sealed class BenchmarkConfig : ManualConfig
    {
        public BenchmarkConfig()
        {
            AddJob(Job.Default
                // .WithRuntime(CoreRuntime.CreateForNewVersion("net10.0", ".NET 10"))
                .WithWarmupCount(5)
                .WithIterationCount(20)
                .WithInvocationCount(1)
                .WithUnrollFactor(1));

            AddDiagnoser(MemoryDiagnoser.Default);
            AddColumn(RankColumn.Arabic);

            WithOrderer(new DefaultOrderer(SummaryOrderPolicy.FastestToSlowest));
        }
    }
}