// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// Which face of the local Kubernetes lab a test needs.
    ///
    /// <para>The lab presents two, on separate routes, and having one up says nothing about the other —
    /// which cost an afternoon on 2026-08-07 when five diagnostics failed on a missing Prometheus forward
    /// while the replica forwards were irrelevant to them, and a sixth failed for the opposite reason.</para>
    /// </summary>
    internal enum LabEndpoint
    {
        /// <summary>
        /// Prometheus, via <c>k8s\monitoring\forward.cmd</c> — read by every guard and calibration
        /// diagnostic. Forwarded on 9090, 9098 and 9099 because the diagnostics default to different
        /// ports.
        /// </summary>
        Prometheus,

        /// <summary>
        /// One local port per <c>overfit-server</c> replica, via <c>k8s\overfit\forward-replicas.cmd</c>.
        /// A service-level forward is not enough: the service is headless and pins one endpoint, so all
        /// traffic lands on one pod and the peer group looks idle.
        /// </summary>
        Replicas,
    }
}
