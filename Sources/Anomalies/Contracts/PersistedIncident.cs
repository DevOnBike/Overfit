// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One open incident, flattened to what has to survive a restart.
    ///
    /// <para><b>Identity and enough to close honestly, not the evidence.</b> Keeping every finding would make
    /// the state file grow with the noisiest cycle and would claim, after a restart, to still hold evidence
    /// gathered by a process that no longer exists. What is kept is who the incident is about, what named it,
    /// how bad it got and when it started — which is what a <c>Resolved</c> row needs and what somebody
    /// closing a ticket asks.</para>
    /// </summary>
    /// <param name="Id">The identity that must not change across the restart.</param>
    /// <param name="FirstSeen">When it opened — the field that makes age real rather than per-window.</param>
    /// <param name="LastSeen">Cycle it was last observed in.</param>
    /// <param name="CyclesSeen">How many cycles have observed it.</param>
    /// <param name="CyclesMissing">Consecutive absences at the time of writing.</param>
    /// <param name="PrimaryKey">Whose problem it is — the key matching is done on.</param>
    /// <param name="SubjectKeys">Every subject it covered, for the overlap test.</param>
    /// <param name="Namespace">Primary subject's namespace.</param>
    /// <param name="Workload">Primary subject's workload.</param>
    /// <param name="ReplicaSet">Primary subject's ReplicaSet, where known.</param>
    /// <param name="Pod">Primary subject's pod, empty for a common-mode incident.</param>
    /// <param name="Node">Primary subject's node, where known.</param>
    /// <param name="Signal">What named the incident.</param>
    /// <param name="Class">Where that signal sits between cause and consequence.</param>
    /// <param name="Severity">Peak severity reached.</param>
    /// <param name="Start">Start of the last evaluated window.</param>
    /// <param name="End">End of it.</param>
    /// <param name="Subjects">Distinct subjects it covered.</param>
    /// <param name="Signals">Distinct signals it covered.</param>
    /// <param name="Summary">The one-line description a consumer already saw.</param>
    /// <param name="Novelty">
    /// Whether the primary finding was a standing deviation. Last, and defaulted, so a payload written before
    /// this existed reads unchanged and comes back as <see cref="NoveltyKind.New"/> — the fail-open answer.
    /// </param>
    public readonly record struct PersistedIncident(
        long Id,
        DateTimeOffset FirstSeen,
        DateTimeOffset LastSeen,
        int CyclesSeen,
        int CyclesMissing,
        string PrimaryKey,
        IReadOnlyList<string> SubjectKeys,
        string Namespace,
        string Workload,
        string ReplicaSet,
        string Pod,
        string Node,
        string Signal,
        SignalClass Class,
        double Severity,
        DateTimeOffset Start,
        DateTimeOffset End,
        int Subjects,
        int Signals,
        string Summary,
        NoveltyKind Novelty = NoveltyKind.New);
}
