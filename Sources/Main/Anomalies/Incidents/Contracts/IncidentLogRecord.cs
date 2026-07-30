// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
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
    /// <param name="IncidentKey">Joins a finding row to its incident row within one reporting cycle.
    /// <b>Not stable across cycles</b> — the pipeline is stateless, so this is a correlation key, not an
    /// incident identity. Anything building a lifecycle on it will double-count.</param>
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
    public readonly record struct IncidentLogRecord(
        int IncidentKey,
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
        string Message)
    {
        /// <summary>How long the behaviour has been running.</summary>
        public TimeSpan Duration => End - Start;

        /// <summary>
        /// Whether this row blames a specific pod. False for common-mode rows, which are about the
        /// deployment — see the remarks on <see cref="IncidentLogRecord"/>.
        /// </summary>
        public bool NamesAPod => Pod.Length > 0;
    }
}
