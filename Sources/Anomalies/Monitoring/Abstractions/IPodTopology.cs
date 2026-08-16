// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Abstractions
{
    /// <summary>
    /// Where each pod sits — the coordinates the grouper relates findings by.
    ///
    /// <para><b>Getting this wrong is not a cosmetic error.</b> The grouper scores <c>SameWorkload</c> at 0.7
    /// against a 0.35 threshold, so telling it that four pods share a workload merges every finding on all of
    /// them into one incident. That happened: the guard stamped one configured workload onto every pod, and
    /// the lab's deliberately degraded replica — its own Deployment — was absorbed into a single group with
    /// the healthy three. The grouper was right; it was being lied to.</para>
    ///
    /// <para><b>A snapshot, deliberately.</b> Resolution is synchronous because a detection cycle should not
    /// wait on the network in the middle of grouping; whoever drives the loop refreshes the snapshot between
    /// cycles instead.</para>
    /// </summary>
    public interface IPodTopology
    {
        /// <summary>
        /// Resolves a pod. Returns <c>false</c> when nothing is known about it, which the caller must treat
        /// as "fall back" rather than as "it belongs to nothing" — an unresolved pod given an empty workload
        /// would share that emptiness with every other unresolved pod and merge with all of them.
        /// </summary>
        bool TryResolve(string pod, out PodPlacement placement);
    }
}
