// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// Global parallel execution settings for CPU BeastMode.
    ///
    /// Default:
    /// - use all logical processors
    /// - ensure ThreadPool minimum worker count is at least ProcessorCount
    ///
    /// Do not mutate Options.MaxDegreeOfParallelism directly from random call sites.
    /// If runtime configurability is needed later, add a controlled setter here.
    /// </summary>
    public static class OverfitParallel
    {
        public static readonly int MaxDegreeOfParallelism = Environment.ProcessorCount;

        public static readonly ParallelOptions Options = new()
        {
            MaxDegreeOfParallelism = MaxDegreeOfParallelism
        };

        static OverfitParallel()
        {
            ThreadPool.GetMinThreads(out var minWorkerThreads, out var minCompletionPortThreads);

            if (minWorkerThreads < MaxDegreeOfParallelism)
            {
                ThreadPool.SetMinThreads(MaxDegreeOfParallelism, minCompletionPortThreads);
            }
        }
    }
}