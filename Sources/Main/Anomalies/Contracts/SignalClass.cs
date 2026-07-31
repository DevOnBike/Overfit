// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// How close a signal sits to a cause, as opposed to a consequence. The grouper uses this to decide which
    /// finding in an incident gets shown first — declaration order is the ranking, so members must stay
    /// ordered cause-first.
    ///
    /// <para><b>This is a heuristic about where to look, not causal inference.</b> Nothing here establishes
    /// that the infrastructure event caused the symptom; a relative method cannot. It encodes only that an
    /// engineer handed "pod was CPU-throttled" and "p95 latency doubled" for the same pod in the same minute
    /// should be shown the throttling first, because it is the one they can act on.</para>
    /// </summary>
    public enum SignalClass
    {
        /// <summary>
        /// The platform acting on the workload: restarts, OOMKills, CFS throttling, eviction, node pressure,
        /// image pull failures. Load-independent, and normally the actionable end of an incident.
        /// </summary>
        Infrastructure,

        /// <summary>
        /// The workload's own consumption: memory, CPU, thread-pool queue, GC pressure, connection counts.
        /// Between cause and symptom — a leak is here, and it explains the symptoms above it.
        /// </summary>
        Resource,

        /// <summary>
        /// What the user experiences: latency, error rate, throughput, saturation of a request path. Usually
        /// the first thing noticed and the last thing worth investigating, because it is downstream of
        /// everything else.
        /// </summary>
        Symptom
    }
}
