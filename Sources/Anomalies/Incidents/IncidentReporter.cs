// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// Flattens grouped incidents into <see cref="IncidentLogRecord"/> rows and hands them to a sink.
    ///
    /// <para>The one place the tree becomes rows, so a log, a metrics exporter and a table cannot disagree
    /// about what a row is. Every backend then maps the same field names, and a saved query keeps working
    /// when a new one is added.</para>
    ///
    /// <para>Rows are built into pooled scratch and passed as a span; nothing is retained. A sink that keeps
    /// them copies them, which is stated on <see cref="IIncidentSink.Report"/>.</para>
    /// </summary>
    public static class IncidentReporter
    {
        /// <summary>
        /// Upper bound on rows per call: one per incident plus one per finding. Guards the pooled buffer
        /// against a cycle that has stopped filtering, which is the condition
        /// <see cref="IncidentGrouper.MaxFindingsPerCall"/> already names.
        /// </summary>
        public const int MaxRowsPerCall = 2 * IncidentGrouper.MaxFindingsPerCall;

        /// <summary>
        /// Reports every incident in <paramref name="incidents"/> as one incident row plus one row per
        /// finding, joined by <see cref="IncidentLogRecord.IncidentKey"/>.
        /// </summary>
        /// <param name="incidents">A cycle's output, as returned by <see cref="IncidentTracker.Observe"/> —
        /// tracked rather than raw, so the lifecycle state reaches the sink. Reporting raw groups instead
        /// loses the tracker entirely at the last boundary: every cycle looks like a fresh incident.</param>
        /// <param name="sink">Destination.</param>
        /// <param name="suppressedBy">
        /// The maintenance reason to stamp on every row, or empty. Passed down rather than decided here:
        /// whether a cycle fell inside a declared window is the guard's knowledge, and the reporter's job is
        /// to carry it to the sink so a host can route on it instead of paging.
        /// </param>
        /// <returns>How many rows were reported.</returns>
        public static int Report(
            IReadOnlyList<TrackedIncident> incidents, IIncidentSink sink, string suppressedBy = "")
        {
            ArgumentNullException.ThrowIfNull(incidents);
            ArgumentNullException.ThrowIfNull(sink);
            ArgumentNullException.ThrowIfNull(suppressedBy);

            if (incidents.Count == 0)
            {
                return 0;
            }

            var needed = 0;

            for (var i = 0; i < incidents.Count; i++)
            {
                needed += 1 + incidents[i].Incident.Findings.Count;
            }

            if (needed > MaxRowsPerCall)
            {
                throw new InvalidOperationException(
                    $"{needed} rows exceeds the per-call bound of {MaxRowsPerCall}. A cycle this large means a "
                    + "detector has stopped filtering; raise its thresholds or report in batches.");
            }

            using var scratch = new PooledBuffer<IncidentLogRecord>(needed, clearMemory: false);
            var rows = scratch.Span[..needed];
            var written = 0;

            for (var i = 0; i < incidents.Count; i++)
            {
                var tracked = incidents[i];
                var incident = tracked.Incident;
                var subject = incident.Primary.Subject;

                rows[written] = new IncidentLogRecord(
                    IncidentKey: i,
                    IncidentId: tracked.Id,
                    State: tracked.State,
                    Kind: IncidentLogRecordKind.Incident,
                    Namespace: subject.Namespace,
                    Workload: subject.Workload,
                    Pod: subject.Pod,
                    Node: subject.Node,
                    Signal: incident.Primary.Signal,
                    Class: incident.Primary.Class,
                    Severity: incident.PeakSeverity,
                    Start: incident.Start,
                    End: incident.End,
                    Subjects: incident.AffectedSubjects,
                    Signals: incident.DistinctSignals,
                    Message: incident.Summary,

                    // Only on the incident row. A finding row is one line of evidence and repeating the whole
                    // explanation on each would bloat every log by the square of the group size.
                    Narrative: IncidentNarrative.Describe(incident),
                    SuppressedBy: suppressedBy);

                written++;

                for (var f = 0; f < incident.Findings.Count; f++)
                {
                    var finding = incident.Findings[f];

                    rows[written] = new IncidentLogRecord(
                        IncidentKey: i,
                        IncidentId: tracked.Id,
                        State: tracked.State,
                        Kind: IncidentLogRecordKind.Finding,
                        Namespace: finding.Subject.Namespace,
                        Workload: finding.Subject.Workload,
                        Pod: finding.Subject.Pod,
                        Node: finding.Subject.Node,
                        Signal: finding.Signal,
                        Class: finding.Class,
                        Severity: finding.Severity,
                        Start: finding.Start,
                        End: finding.End,
                        Subjects: 1,
                        Signals: 1,
                        Message: finding.Reason,
                        Narrative: "",
                        SuppressedBy: suppressedBy);

                    written++;
                }
            }

            sink.Report(rows);

            return written;
        }
    }
}
