// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Abstractions
{
    /// <summary>
    /// A topology whose snapshot can be brought up to date between cycles.
    ///
    /// <para>Split from <see cref="IPodTopology"/> so that resolution stays synchronous — a detection cycle
    /// must not wait on the network while grouping — while the refresh, which is I/O, sits where the loop can
    /// schedule it.</para>
    /// </summary>
    public interface IRefreshablePodTopology : IPodTopology
    {
        /// <summary>
        /// Re-reads the cluster. Returns how many pods were resolved, or <b>-1</b> when the refresh failed or
        /// matched nothing — in which case the previous snapshot must stand.
        ///
        /// <para>Returning -1 rather than throwing is the contract because the caller's correct response is
        /// to carry on with stale coordinates and say so. An empty topology is not neutral: it gives every
        /// pod the same blank workload, and the grouper then merges the entire namespace into one incident.</para>
        /// </summary>
        Task<int> RefreshAsync(CancellationToken ct = default);
    }
}
