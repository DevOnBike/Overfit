// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Incidents.Contracts;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// The tracker's open incidents, as a line of text each.
    ///
    /// <para><b>Hand-rolled rather than JSON, for the same reason the ONNX parser is.</b> The library is
    /// built Native-AOT-clean and reflection-based serialisation is banned here; adding these types to the
    /// source-generated context would work but buys nothing for a format this small. Tab-separated fields,
    /// one incident per line, a version marker first — readable in a terminal when somebody is trying to
    /// work out why the guard reopened everything.</para>
    ///
    /// <para><b>What is persisted is identity, not evidence.</b> Enough to keep an incident the same incident
    /// across a restart and to emit an honest <c>Resolved</c> row later: who it is about, what named it, how
    /// bad it got, when it started. The individual findings are not kept — the consumer has already seen
    /// them, and a restored incident says plainly that its evidence predates the restart rather than
    /// pretending to still hold it.</para>
    ///
    /// <para><b>An unreadable line is skipped, not fatal.</b> A state file that has been truncated or
    /// hand-edited should cost the incidents it can no longer describe, not the whole run.</para>
    /// </summary>
    public static class IncidentStateFormat
    {
        private const string Header = "overfit-incident-state\tv1";
        private const char Separator = '\t';

        /// <summary>Renders open incidents plus the next identifier to hand out.</summary>
        public static string Write(IReadOnlyList<PersistedIncident> incidents, long nextId)
        {
            ArgumentNullException.ThrowIfNull(incidents);

            var text = new StringBuilder();

            text.Append(Header).Append('\n');
            text.Append(nextId.ToString(CultureInfo.InvariantCulture)).Append('\n');

            for (var i = 0; i < incidents.Count; i++)
            {
                var incident = incidents[i];

                Append(text, incident.Id.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.FirstSeen.UtcTicks.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.LastSeen.UtcTicks.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.CyclesSeen.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.CyclesMissing.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.PrimaryKey);
                Append(text, string.Join('|', incident.SubjectKeys));
                Append(text, incident.Namespace);
                Append(text, incident.Workload);
                Append(text, incident.ReplicaSet);
                Append(text, incident.Pod);
                Append(text, incident.Node);
                Append(text, incident.Signal);
                Append(text, ((int)incident.Class).ToString(CultureInfo.InvariantCulture));
                Append(text, incident.Severity.ToString("R", CultureInfo.InvariantCulture));
                Append(text, incident.Start.UtcTicks.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.End.UtcTicks.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.Subjects.ToString(CultureInfo.InvariantCulture));
                Append(text, incident.Signals.ToString(CultureInfo.InvariantCulture));

                text.Append(Escape(incident.Summary)).Append('\n');
            }

            return text.ToString();
        }

        /// <summary>
        /// Parses what <see cref="Write"/> produced. Returns an empty list and <c>nextId</c> of 1 for
        /// anything it does not recognise, which is the cold start the caller must be able to survive anyway.
        /// </summary>
        public static IReadOnlyList<PersistedIncident> Read(string? state, out long nextId)
        {
            nextId = 1;

            var incidents = new List<PersistedIncident>();

            if (string.IsNullOrWhiteSpace(state))
            {
                return incidents;
            }

            var lines = state.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            if (lines.Length == 0 || !lines[0].StartsWith(Header, StringComparison.Ordinal))
            {
                // A different version, or not our file at all. Starting cold beats guessing at a layout.
                return incidents;
            }

            if (lines.Length < 2
                || !long.TryParse(lines[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out nextId))
            {
                nextId = 1;

                return incidents;
            }

            for (var i = 2; i < lines.Length; i++)
            {
                if (TryReadLine(lines[i], out var incident))
                {
                    incidents.Add(incident);
                }
            }

            return incidents;
        }

        private static bool TryReadLine(string line, out PersistedIncident incident)
        {
            incident = default;

            var f = line.Split(Separator);

            if (f.Length < 20)
            {
                return false;
            }

            if (!long.TryParse(f[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var id)
                || !long.TryParse(f[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var first)
                || !long.TryParse(f[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var last)
                || !int.TryParse(f[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out var seen)
                || !int.TryParse(f[4], NumberStyles.Integer, CultureInfo.InvariantCulture, out var missing)
                || !int.TryParse(f[13], NumberStyles.Integer, CultureInfo.InvariantCulture, out var cls)
                || !double.TryParse(f[14], NumberStyles.Float, CultureInfo.InvariantCulture, out var severity)
                || !long.TryParse(f[15], NumberStyles.Integer, CultureInfo.InvariantCulture, out var start)
                || !long.TryParse(f[16], NumberStyles.Integer, CultureInfo.InvariantCulture, out var end)
                || !int.TryParse(f[17], NumberStyles.Integer, CultureInfo.InvariantCulture, out var subjects)
                || !int.TryParse(f[18], NumberStyles.Integer, CultureInfo.InvariantCulture, out var signals))
            {
                return false;
            }

            incident = new PersistedIncident(
                id,
                new DateTimeOffset(first, TimeSpan.Zero),
                new DateTimeOffset(last, TimeSpan.Zero),
                seen,
                missing,
                Unescape(f[5]),
                f[6].Length == 0 ? [] : Unescape(f[6]).Split('|'),
                Unescape(f[7]),
                Unescape(f[8]),
                Unescape(f[9]),
                Unescape(f[10]),
                Unescape(f[11]),
                Unescape(f[12]),
                (SignalClass)cls,
                severity,
                new DateTimeOffset(start, TimeSpan.Zero),
                new DateTimeOffset(end, TimeSpan.Zero),
                subjects,
                signals,
                Unescape(f[19]));

            return true;
        }

        private static void Append(StringBuilder text, string value)
        {
            text.Append(Escape(value)).Append(Separator);
        }

        /// <summary>
        /// Tabs and newlines are the record structure, so a summary containing one would otherwise split a
        /// line into two unreadable ones — and detector reason strings are free text.
        /// </summary>
        private static string Escape(string value)
        {
            return value
                .Replace("\\", "\\\\", StringComparison.Ordinal)
                .Replace("\t", "\\t", StringComparison.Ordinal)
                .Replace("\n", "\\n", StringComparison.Ordinal)
                .Replace("\r", "\\r", StringComparison.Ordinal);
        }

        private static string Unescape(string value)
        {
            return value
                .Replace("\\r", "\r", StringComparison.Ordinal)
                .Replace("\\n", "\n", StringComparison.Ordinal)
                .Replace("\\t", "\t", StringComparison.Ordinal)
                .Replace("\\\\", "\\", StringComparison.Ordinal);
        }
    }
}
