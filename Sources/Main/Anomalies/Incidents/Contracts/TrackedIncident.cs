// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
{
    /// <summary>
    /// An incident with an identity that survives across evaluation cycles.
    /// </summary>
    /// <param name="Id">
    /// Stable for as long as the incident is open. Assigned by the <c>IncidentTracker</c> that produced it and
    /// unique within that instance's lifetime — <b>not</b> across restarts, because nothing here persists.
    /// Pair it with a run identifier if it has to be globally unique.
    /// </param>
    /// <param name="State">Opened this cycle, still running, or just closed.</param>
    /// <param name="Incident">
    /// The most recent grouping. For a <see cref="IncidentState.Resolved"/> row this is the last state it was
    /// seen in, not an empty one — a consumer closing a ticket wants to know what it was.
    /// </param>
    /// <param name="FirstSeen">When it opened. With <paramref name="LastSeen"/> this is the real duration,
    /// as opposed to <see cref="Contracts.Incident.Duration"/>, which is only the evaluated window.</param>
    /// <param name="LastSeen">Cycle in which it was last observed.</param>
    /// <param name="CyclesSeen">How many cycles have observed it — a flap shows up here as a low count
    /// against a long span.</param>
    /// <param name="CyclesMissing">Consecutive cycles it has been absent. Non-zero on an
    /// <see cref="IncidentState.Ongoing"/> row means it is inside the grace period rather than steady.</param>
    public readonly record struct TrackedIncident(
        long Id,
        IncidentState State,
        Incident Incident,
        DateTimeOffset FirstSeen,
        DateTimeOffset LastSeen,
        int CyclesSeen,
        int CyclesMissing)
    {
        /// <summary>
        /// How long it has actually been running, across cycles — not the width of one evaluation window.
        /// </summary>
        public TimeSpan Age => LastSeen - FirstSeen;

        /// <summary>
        /// Whether this row is the one and only moment a consumer should notify a human. Every other state
        /// is an update to something already reported.
        /// </summary>
        public bool IsNewlyOpened => State == IncidentState.Opened;
    }
}
