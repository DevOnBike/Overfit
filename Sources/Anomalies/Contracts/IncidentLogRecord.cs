// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One row of structured output: an incident flattened for a log, a metric label set, or a table.
    ///
    /// <para><b>This is the schema, and that is the whole reason it exists separately from
    /// <see cref="Incident"/>.</b> A dashboard query, a log filter and an alert rule are all written against
    /// field <i>names</i>. Once someone has saved a query on <c>Workload</c> and <c>Severity</c>, those names
    /// are an interface — and interning them here means the internal contracts can keep changing shape
    /// without silently breaking a search somebody depends on.</para>
    ///
    /// <para><b>Flat, and deliberately so.</b> An incident is a tree — a group, its findings, each with a
    /// subject — and a log line is not. Nesting it would produce either an unqueryable blob or a JSON string
    /// the reader has to parse back out. So a group becomes one incident row plus one row per finding, joined
    /// by <see cref="IncidentKey"/>.</para>
    ///
    /// <para><b>A pod-less row is not a missing field.</b> Common-mode findings — the deployment as a whole
    /// warming up, being rolled, or serving rising load — carry no pod by design, because naming one would be
    /// a false statement. Consumers must treat empty <see cref="Pod"/> as "about the workload", not as
    /// "unknown pod"; the distinction is the difference between one honest row and N wrong ones.</para>
    /// </summary>
    /// <param name="Narrative">
    /// Multi-line human explanation — what was seen, over which interval, on exactly which objects, what
    /// moved with it, and what the evidence cannot settle. Built by <see cref="Incidents.IncidentNarrative"/>.
    ///
    /// <para><b>Populated on the incident row only</b>, and empty on every finding row. A finding is one
    /// line of evidence; repeating the group's whole explanation on each would grow a log by the square of
    /// the group size for no added information.</para>
    ///
    /// <para>Do not query or alert on this field. It is prose meant for the person who opens the incident,
    /// and it is the one field here whose wording is expected to change; <see cref="Message"/>,
    /// <see cref="Signal"/> and <see cref="Severity"/> are the stable surface.</para>
    /// </param>
    /// <param name="IncidentKey">Joins a finding row to its incident row within one reporting cycle.
    /// <b>Not stable across cycles</b> — use <paramref name="IncidentId"/> for that.</param>
    /// <param name="IncidentId">
    /// Stable identity for as long as the incident is open, from <c>IncidentTracker</c>. Zero when the rows
    /// were produced without tracking.
    /// </param>
    /// <param name="State">
    /// Opened this cycle, still running, or just closed.
    ///
    /// <para><b>Without this the tracker's work does not reach the consumer.</b> A sink that receives the
    /// same incident every cycle with no way to tell a new one from a continuing one logs twelve identical
    /// lines for a one-hour problem — which is precisely the behaviour the tracker exists to remove, arriving
    /// at the last possible boundary. Notify on <see cref="IncidentState.Opened"/>; update on the rest.</para>
    /// </param>
    /// <param name="Kind">Whether this row describes the group or one finding inside it.</param>
    /// <param name="Namespace">Kubernetes namespace.</param>
    /// <param name="Workload">Deployment or StatefulSet the subject belongs to.</param>
    /// <param name="Pod">Pod name, or empty for a row about the workload as a whole.</param>
    /// <param name="Node">Node, where known.</param>
    /// <param name="Signal">Metric or rule name — stable across pods, so it groups in a query.</param>
    /// <param name="Class">Where the signal sits between cause and consequence.</param>
    /// <param name="Severity">Normalised 0…1, comparable across detectors.</param>
    /// <param name="Start">Start of the evaluated window, not the moment of detection.</param>
    /// <param name="End">End of it.</param>
    /// <param name="Subjects">Distinct subjects in the group; 1 for a finding row.</param>
    /// <param name="Signals">Distinct signals in the group; 1 for a finding row.</param>
    /// <param name="Message">The incident summary, or the detector's reason for a finding row.</param>
    /// <param name="SuppressedBy">
    /// The operator's reason when a declared maintenance window covered this, empty otherwise. <b>The row
    /// is still emitted</b> — deleting the evidence to keep the log tidy removes exactly the record somebody
    /// comes back for after a failed deploy. What this changes is routing: a host must not page on it.
    /// </param>
    public readonly record struct IncidentLogRecord(
        int IncidentKey,
        long IncidentId,
        IncidentState State,
        IncidentLogRecordKind Kind,
        string Namespace,
        string Workload,
        string Pod,
        string Node,
        string Signal,
        SignalClass Class,
        double Severity,
        DateTimeOffset Start,
        DateTimeOffset End,
        int Subjects,
        int Signals,
        string Message,
        string Narrative = "",
        string SuppressedBy = "")
    {
        /// <summary>
        /// Whether a declared maintenance window covered this. The row is still emitted, because an operator
        /// looking at a failed deploy wants to know what the guard saw during it — deleting the evidence to
        /// keep the log tidy removes exactly the record they came for. What the flag changes is routing: a
        /// host must not page on it.
        /// </summary>
        public bool IsSuppressed => SuppressedBy.Length > 0;

        /// <summary>How long the behaviour has been running.</summary>
        public TimeSpan Duration => End - Start;

        /// <summary>
        /// Whether this row is the one moment a consumer should notify a human. Every other state is an
        /// update to something already reported.
        /// </summary>
        public bool IsNews => State is IncidentState.Opened or IncidentState.Resolved;

        /// <summary>
        /// Whether this row blames a specific pod. False for common-mode rows, which are about the
        /// deployment — see the remarks on <see cref="IncidentLogRecord"/>.
        /// </summary>
        public bool NamesAPod => Pod.Length > 0;
    }
}
