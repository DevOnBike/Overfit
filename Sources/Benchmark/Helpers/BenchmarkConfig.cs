// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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

            // Rates are derived in-repo from the benchmark's own declared WorkAmount; classes that declare
            // none get empty cells. See WorkAmount for why this is not computed in an external script.
            AddColumn(ThroughputColumn.Teraflops);
            AddColumn(ThroughputColumn.Gigabytes);

            WithOrderer(new DefaultOrderer(SummaryOrderPolicy.FastestToSlowest));
        }
    }
}