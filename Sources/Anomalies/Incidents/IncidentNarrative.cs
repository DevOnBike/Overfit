// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// Writes out an incident the way a person needs to read it: what was seen, over what interval, on
    /// exactly which objects, what else moved with it, and what the evidence cannot settle.
    ///
    /// <para><b>Why the one-line summary is not enough.</b> <see cref="Incident.Summary"/> exists to fit in a
    /// log line and it does that job. But an operator woken by it has to answer three questions the line does
    /// not address: <i>which object do I open</i>, <i>what interval do I look at</i>, and <i>how sure is
    /// this</i>. A shadow run made the gap concrete — every row read "Series rose by 10.9% of typical", which
    /// is true, machine-checkable, and does not tell the reader that the shape in question is a GC sawtooth
    /// rather than a leak.</para>
    ///
    /// <para><b>Identifiers are spelled out rather than summarised.</b> A pod name alone is not enough to act
    /// on: the namespace decides which cluster context to use, the workload is the level a human reasons at,
    /// the ReplicaSet distinguishes one rollout's pods from the previous rollout's, and the node is what turns
    /// "three replicas are slow" into "three replicas on one node are slow". Each is printed when known and
    /// silently skipped when not, because an empty field invites the reader to guess.</para>
    ///
    /// <para><b>The interval is the observation window, not the incident's age.</b> It states when the
    /// behaviour was <i>seen</i>, which is what a dashboard query needs. How long the incident has been open
    /// is the tracker's business and lives on <c>TrackedIncident</c>.</para>
    ///
    /// <para><b>The last section is the honest one.</b> A relative method has no external reference, so "this
    /// replica regressed" and "the others improved" produce identical evidence; a symptom signal says
    /// something hurts and not why. Printing that next to the finding costs three lines and stops the reader
    /// from over-reading a number — which is the failure mode that makes people stop trusting a detector
    /// altogether.</para>
    ///
    /// <para>Reporting code, called once per incident per cycle. Not a hot path, and it allocates a string on
    /// purpose.</para>
    /// </summary>
    public static class IncidentNarrative
    {
        /// <summary>Related findings listed in full before the rest are counted.</summary>
        private const int MaxListedFindings = 6;

        /// <summary>
        /// A multi-line explanation of one incident. Never empty.
        /// </summary>
        /// <param name="incident">The grouped incident, findings already ordered cause-first.</param>
        public static string Describe(Incident incident)
        {
            ArgumentNullException.ThrowIfNull(incident);

            var text = new StringBuilder(600);
            var primary = incident.Primary;

            AppendHeadline(text, incident, primary);
            AppendWhen(text, incident);
            AppendWhere(text, incident);
            AppendScope(text, incident);
            AppendRelated(text, incident);
            AppendCaveats(text, incident, primary);

            return text.ToString();
        }

        private static void AppendHeadline(StringBuilder text, Incident incident, in SignalFinding primary)
        {
            text.Append("WHAT   ")
                .Append(primary.Signal)
                .Append(" on ")
                .Append(primary.Subject.Label)
                .Append("\n       ")
                .Append(primary.Reason)
                .Append('\n');
        }

        private static void AppendWhen(StringBuilder text, Incident incident)
        {
            // Absolute UTC, because this is copied into a dashboard query or a kubectl --since more often
            // than it is read as prose, and a relative "15 minutes ago" is wrong the moment it is stored.
            text.Append("WHEN   ")
                .Append(incident.Start.UtcDateTime.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture))
                .Append("Z to ")
                .Append(incident.End.UtcDateTime.ToString("HH:mm:ss", CultureInfo.InvariantCulture))
                .Append("Z (")
                .Append(Humanise(incident.Duration))
                .Append(" observed)\n");
        }

        private static void AppendWhere(StringBuilder text, Incident incident)
        {
            var subject = incident.Primary.Subject;

            text.Append("WHERE  ");

            var wrote = false;

            wrote |= AppendField(text, "namespace", subject.Namespace, wrote);
            wrote |= AppendField(text, "workload", subject.Workload, wrote);
            wrote |= AppendField(text, "replicaset", subject.ReplicaSet, wrote);

            if (subject.Pod.Length > 0)
            {
                text.Append(wrote ? "\n       " : string.Empty).Append("pod ").Append(subject.Pod);
                wrote = true;
            }

            if (subject.Node.Length > 0)
            {
                text.Append(wrote ? " on node " : "node ").Append(subject.Node);
                wrote = true;
            }

            if (!wrote)
            {
                text.Append("(no identifiers on the subject)");
            }

            if (subject.Pod.Length == 0)
            {
                // Said explicitly: an empty pod is a statement, not a gap. Common-mode findings are about the
                // deployment, and naming a replica would be a false claim rather than a missing one.
                text.Append("\n       about the workload as a whole — no single replica is responsible");
            }

            text.Append('\n');
        }

        private static void AppendScope(StringBuilder text, Incident incident)
        {
            text.Append("SCOPE  ")
                .Append(incident.Findings.Count)
                .Append(incident.Findings.Count == 1 ? " finding across " : " findings across ")
                .Append(incident.AffectedSubjects)
                .Append(incident.AffectedSubjects == 1 ? " object and " : " objects and ")
                .Append(incident.DistinctSignals)
                .Append(incident.DistinctSignals == 1 ? " signal; " : " signals; ")
                .Append("peak severity ")
                .Append(incident.PeakSeverity.ToString("F2", CultureInfo.InvariantCulture))
                .Append('\n');
        }

        private static void AppendRelated(StringBuilder text, Incident incident)
        {
            if (incident.Findings.Count <= 1)
            {
                return;
            }

            text.Append("ALSO   ");

            var listed = 0;

            for (var i = 0; i < incident.Findings.Count; i++)
            {
                // Index 0 is the primary and has already been stated in full.
                if (i == 0)
                {
                    continue;
                }

                if (listed == MaxListedFindings)
                {
                    text.Append("\n       and ")
                        .Append(incident.Findings.Count - 1 - listed)
                        .Append(" more");

                    break;
                }

                var finding = incident.Findings[i];

                text.Append(listed == 0 ? string.Empty : "\n       ")
                    .Append(finding.Signal)
                    .Append(" on ")
                    .Append(finding.Subject.Label)
                    .Append(" — ")
                    .Append(finding.Reason);

                listed++;
            }

            text.Append('\n');
        }

        private static void AppendCaveats(StringBuilder text, Incident incident, in SignalFinding primary)
        {
            text.Append("CAVEAT ");

            var wrote = false;

            if (primary.Class == SignalClass.Symptom)
            {
                text.Append("this is a symptom — it says something hurts, not why. Look for a cause among "
                            + "the resource and infrastructure signals on the same objects.");
                wrote = true;
            }

            if (primary.Class == SignalClass.Infrastructure)
            {
                text.Append("this sits close to a cause, so treat it as the thing to fix rather than as "
                            + "evidence of something further upstream.");
                wrote = true;
            }

            if (incident.AffectedSubjects > 1)
            {
                text.Append(wrote ? "\n       " : string.Empty)
                    .Append(incident.AffectedSubjects)
                    .Append(" objects moved together, which points at something they share — a node, a "
                            + "dependency, a rollout — rather than at any one of them.");
                wrote = true;
            }

            if (incident.Primary.Subject.Pod.Length > 0 && incident.AffectedSubjects == 1)
            {
                text.Append(wrote ? "\n       " : string.Empty)
                    .Append("a peer comparison is relative and has no external reference: \"this replica got "
                            + "worse\" and \"its siblings got better\" leave identical evidence. Confirm "
                            + "against this workload's own history before concluding which happened.");
                wrote = true;
            }

            if (!wrote)
            {
                text.Append("none.");
            }

            text.Append('\n');
        }

        private static bool AppendField(StringBuilder text, string name, string value, bool wrote)
        {
            if (value.Length == 0)
            {
                return false;
            }

            text.Append(wrote ? " / " : string.Empty).Append(name).Append(' ').Append(value);

            return true;
        }

        /// <summary>
        /// Whole units, because a window measured to the second reads as false precision. Invariant, like
        /// the timestamps above it: <c>{x:F1}</c> is <c>1.5</c> here and <c>1,5</c> on a European desktop,
        /// and this text is grepped.
        /// </summary>
        private static string Humanise(TimeSpan span)
        {
            if (span.TotalMinutes < 1.0)
            {
                return string.Create(CultureInfo.InvariantCulture, $"{span.TotalSeconds:F0}s");
            }

            if (span.TotalHours < 1.0)
            {
                return string.Create(CultureInfo.InvariantCulture, $"{span.TotalMinutes:F0} min");
            }

            return string.Create(CultureInfo.InvariantCulture, $"{span.TotalHours:F1} h");
        }
    }
}
