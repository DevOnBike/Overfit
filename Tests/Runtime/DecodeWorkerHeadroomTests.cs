// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Runtime
{
    /// <summary>
    /// Guards the decode pool's headroom invariant. The pool SPINS, so if it takes every CPU there is no core
    /// left for the dispatcher and decode throughput collapses. Measured on Qwen-3B Q4_K_M (best-of-3):
    /// 4 CPUs 5.91 → 9.62 tok/s, 8 CPUs 11.69 → 18.83, 10 CPUs 12.13 → 21.38 — i.e. taking one worker away is
    /// worth <b>+61…+76 %</b> on small machines. The previous default (<c>Min(procCount, 10)</c>) put every box
    /// with ≤10 logical CPUs into that cliff. This is a one-line invariant protecting a ~60 % regression, so it
    /// gets a test rather than a comment.
    /// </summary>
    public sealed class DecodeWorkerHeadroomTests
    {
        [Fact]
        public void DecodeMaxWorkers_LeavesAtLeastOneCpuForTheDispatcher()
        {
            var cpus = Environment.ProcessorCount;
            if (cpus <= 1)
            {
                return;   // nothing to leave headroom from
            }

            Assert.InRange(OverfitParallel.DecodeMaxWorkers, 1, cpus - 1);
        }

        [Fact]
        public void DecodeMaxWorkers_IsPositive()
        {
            // The Max(1, …) floor must survive even a single-CPU box, or the pool would spawn zero workers.
            Assert.True(OverfitParallel.DecodeMaxWorkers >= 1);
        }
    }
}
