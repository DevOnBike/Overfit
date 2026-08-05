// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Incidents.Abstractions
{
    /// <summary>
    /// Where a reporting cycle's output goes: a log, a metrics endpoint, a store, a test.
    ///
    /// <para><b>Rows, not objects.</b> The sink receives <see cref="IncidentLogRecord"/> — a flat schema with
    /// stable field names — rather than the <see cref="Incident"/> tree, so a logging backend, a Prometheus
    /// exporter and a table writer all map the same field set without each inventing its own flattening. See
    /// <see cref="IncidentLogRecord"/> for why the schema is separate from the contracts.</para>
    ///
    /// <para><b>Implementations must not throw.</b> A monitoring guard that falls over because its log
    /// destination is unreachable has replaced the problem it was bought to detect with one of its own.
    /// Catch, and if it matters, count.</para>
    ///
    /// <para><b>Deliberately not async.</b> The alternative is an <c>await</c> in the middle of a detection
    /// cycle, and the two things a sink actually does — write a log line, increment a counter — are
    /// synchronous. A sink that genuinely needs I/O should enqueue and drain on its own schedule rather than
    /// make the detector wait on the network.</para>
    /// </summary>
    public interface IIncidentSink
    {
        /// <summary>
        /// Reports one cycle's rows. The span is valid only for the duration of the call — a sink that keeps
        /// rows must copy them.
        /// </summary>
        void Report(ReadOnlySpan<IncidentLogRecord> rows);
    }
}
