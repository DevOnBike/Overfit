// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;
using Microsoft.Extensions.Logging;

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// Writes a detection cycle to <see cref="ILogger"/> as structured events.
    ///
    /// <para><b>Structured, not formatted.</b> Every field is a named message-template parameter, so Loki,
    /// Seq, Splunk and Elastic index them as fields rather than as a sentence somebody later regrets having
    /// to parse. The rendered text exists for a human tailing a console; the fields are the product.</para>
    ///
    /// <para><b>This lives here, and not in the library, on purpose.</b> <c>DevOnBike.Overfit</c> ships with
    /// one runtime dependency and adding <c>Microsoft.Extensions.Logging.Abstractions</c> would put it — and
    /// <c>DependencyInjection.Abstractions</c> behind it — into the public graph of every consumer, including
    /// those embedding the engine with no logging at all. The library defines the schema
    /// (<see cref="IncidentLogRecord"/>) and the contract (<see cref="IIncidentSink"/>); this is the twenty
    /// lines that bind them to one backend, in a project that already has it.</para>
    ///
    /// <para><b>Information, not Warning, and that is a measured position rather than timidity.</b> In most
    /// deployments a log level is a routing decision — Warning and above reaches somebody. On a healthy
    /// synthetic population this guard still produces tens of incidents a day, and on the cluster lab it
    /// produces false findings on healthy replicas in a twelve-minute window. Emitting those at Warning would
    /// train the first operator who sees them to filter the channel out, and that is not recoverable. Raise
    /// <see cref="IncidentLogOptions.Level"/> when the false-positive rate has been measured on the cluster
    /// it will run against — see the shadow-mode note in <c>docs/aiops-detection-pipeline.md</c>.</para>
    ///
    /// <para><b>Findings are separated from incidents deliberately.</b> An incident is what somebody might
    /// act on; a finding is evidence. Counting the wrong one has already produced a wrong conclusion in this
    /// project — 1193 findings were 254 incidents — so they carry different event IDs and can be filtered
    /// apart.</para>
    ///
    /// <para>Uses <see cref="LoggerMessage"/> delegates: the template is parsed once at startup rather than
    /// on every call, there is no boxing of the value-typed arguments, and a disabled level costs a branch
    /// instead of an allocation.</para>
    /// </summary>
    public sealed class LoggerIncidentSink : IIncidentSink
    {
        /// <summary>Anomaly guard event IDs, kept together so they can be filtered as a block.</summary>
        private const int IncidentEventId = 5001;
        private const int FindingEventId = 5002;
        private const int CommonModeEventId = 5003;
        private const int ResolvedEventId = 5007;

        // Built per instance rather than statically, because LoggerMessage.Define bakes the level into the
        // delegate and the level is the one thing a deployment genuinely needs to change. The templates are
        // still parsed once — at construction, not per call, which is the cost the pattern exists to avoid.
        private readonly Action<ILogger, string, string, string, double, int, Exception?> _incident;
        private readonly Action<ILogger, string, string, string, double, Exception?> _commonMode;
        private readonly Action<ILogger, string, string, SignalClass, double, string, Exception?> _finding;
        private readonly Action<ILogger, long, string, string, Exception?> _resolved;

        private readonly ILogger _logger;
        private readonly IncidentLogOptions _options;

        public LoggerIncidentSink(ILogger<LoggerIncidentSink> logger, IncidentLogOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(logger);

            _logger = logger;
            _options = options ?? IncidentLogOptions.Shadow;

            var level = _options.Level;

            _incident = LoggerMessage.Define<string, string, string, double, int>(
                level,
                new EventId(IncidentEventId, "AnomalyIncident"),
                "Anomaly incident in {Namespace}/{Workload}: {Summary} "
                + "(severity {Severity}, {Subjects} subjects)");

            // Common mode gets its own event because it is a different claim: the deployment moved, no
            // replica is accused. Emitted as an ordinary finding with an empty pod it would read as a
            // finding with a missing field, and "unknown pod" is exactly the wrong reading.
            _commonMode = LoggerMessage.Define<string, string, string, double>(
                level,
                new EventId(CommonModeEventId, "AnomalyCommonMode"),
                "Deployment-wide movement in {Namespace}/{Workload} on {Signal} "
                + "(severity {Severity}) — no individual replica is implicated");

            _resolved = LoggerMessage.Define<long, string, string>(
                level,
                new EventId(ResolvedEventId, "AnomalyIncidentResolved"),
                "Anomaly incident {IncidentId} in {Namespace}/{Workload} has closed");

            _finding = LoggerMessage.Define<string, string, SignalClass, double, string>(
                _options.FindingLevel,
                new EventId(FindingEventId, "AnomalyFinding"),
                "Anomaly finding on {Pod}: {Signal} [{Class}] severity {Severity} — {Reason}");
        }

        /// <inheritdoc/>
        public void Report(ReadOnlySpan<IncidentLogRecord> rows)
        {
            // A sink must not throw: a guard that falls over because its log destination is unhappy has
            // replaced the problem it was bought to detect with one of its own.
            try
            {
                Write(rows);
            }
            catch (Exception ex)
            {
                _logger.Log(LogLevel.Error, new EventId(IncidentEventId, "AnomalySinkFailed"), ex,
                    "Anomaly incident reporting failed; detection continues.");
            }
        }

        private void Write(ReadOnlySpan<IncidentLogRecord> rows)
        {
            for (var i = 0; i < rows.Length; i++)
            {
                var row = rows[i];

                if (row.Kind == IncidentLogRecordKind.Incident)
                {
                    // Only state changes reach the log. An Ongoing row is the same problem the operator was
                    // already told about, and emitting one every cycle is the twelve-notifications-per-hour
                    // behaviour the tracker exists to remove — arriving at the last boundary instead of the
                    // first.
                    if (row.State == IncidentState.Ongoing)
                    {
                        continue;
                    }

                    if (row.State == IncidentState.Resolved)
                    {
                        _resolved(_logger, row.IncidentId, row.Namespace, row.Workload, null);

                        continue;
                    }

                    _incident(_logger, row.Namespace, row.Workload, row.Message, row.Severity,
                        row.Subjects, null);

                    continue;
                }

                // Evidence follows its incident: repeating it for an unchanged one is noise.
                if (row.State == IncidentState.Ongoing)
                {
                    continue;
                }

                if (!_options.IncludeFindings)
                {
                    continue;
                }

                if (!row.NamesAPod)
                {
                    _commonMode(_logger, row.Namespace, row.Workload, row.Signal, row.Severity, null);

                    continue;
                }

                _finding(_logger, row.Pod, row.Signal, row.Class, row.Severity, row.Message, null);
            }
        }
    }
}
