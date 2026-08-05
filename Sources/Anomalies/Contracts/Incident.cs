// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Findings that appear to describe one event, presented as one thing to look at.
    ///
    /// <para>This is the unit the product is actually selling. Detecting that p95 doubled is a recording rule
    /// anybody can write; the work is arriving at "one pod, throttled, and here are the four downstream
    /// symptoms it explains" instead of five pages that each say something true.</para>
    /// </summary>
    public sealed class Incident
    {
        internal Incident(
            IReadOnlyList<SignalFinding> findings,
            SignalFinding primary,
            DateTimeOffset start,
            DateTimeOffset end,
            int affectedSubjects,
            int distinctSignals,
            string summary)
        {
            Findings = findings;
            Primary = primary;
            Start = start;
            End = end;
            AffectedSubjects = affectedSubjects;
            DistinctSignals = distinctSignals;
            Summary = summary;
        }

        /// <summary>
        /// Every finding in the group, ordered cause-first: by <see cref="SignalClass"/>, then by how early
        /// the behaviour started, then by severity. The first element is <see cref="Primary"/>.
        /// </summary>
        public IReadOnlyList<SignalFinding> Findings
        {
            get;
        }

        /// <summary>Where to look first. See <see cref="SignalClass"/> for why this is a heuristic.</summary>
        public SignalFinding Primary
        {
            get;
        }

        /// <summary>Earliest start across the group.</summary>
        public DateTimeOffset Start
        {
            get;
        }

        /// <summary>Latest end across the group.</summary>
        public DateTimeOffset End
        {
            get;
        }

        /// <summary>Distinct pods (or workloads, for pod-less findings) involved.</summary>
        public int AffectedSubjects
        {
            get;
        }

        /// <summary>Distinct signal names involved — the breadth that separates "one metric is odd" from
        /// "this thing is on fire".</summary>
        public int DistinctSignals
        {
            get;
        }

        /// <summary>One-line description, built from the primary finding and the shape of the group.</summary>
        public string Summary
        {
            get;
        }

        /// <summary>Highest severity in the group.</summary>
        public double PeakSeverity
        {
            get
            {
                var peak = 0.0;

                for (var i = 0; i < Findings.Count; i++)
                {
                    if (Findings[i].Severity > peak)
                    {
                        peak = Findings[i].Severity;
                    }
                }

                return peak;
            }
        }

        /// <summary>How long the incident has been running.</summary>
        public TimeSpan Duration => End - Start;
    }
}
