// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// A channel that has been bound, has reported on every cycle, and whose value has <b>never once
    /// changed</b> across the whole calibration history.
    ///
    /// <para><b>Why this needed its own concept rather than being folded into blindness.</b> A channel with
    /// no series is counted as blind, appears on every cycle line, and an operator can see it. A channel
    /// bound to a series that never varies reports a number, satisfies every coverage check, contributes to
    /// no finding, and is indistinguishable from a healthy quiet signal at every layer above. Two were found
    /// on 2026-08-08 within hours of each other, both months old:</para>
    ///
    /// <list type="bullet">
    /// <item><c>OomEventsRate</c> was bound to <c>container_oom_events_total</c>, which cAdvisor leaves at
    /// zero on this runtime — measured against a pod Kubernetes reported as OOMKilled, and across 43 series
    /// cluster-wide over an hour the only distinct value present was 0.0.</item>
    /// <item>The three latency channels returned <c>25 + q × 25</c> — 37.50, 48.75, 49.75 — because every
    /// request fell in one histogram bucket, so <c>histogram_quantile</c> was interpolating geometry rather
    /// than reporting measurement. Three channels carrying one bit.</item>
    /// </list>
    ///
    /// <para><b>The two are not equally conclusive, and pretending otherwise would make this unusable.</b>
    /// See <see cref="IsConclusive"/>: a constant NON-ZERO value cannot be produced by a healthy quiet
    /// signal, so it is a defect. A constant zero is genuinely ambiguous — a cluster that simply had no OOM
    /// kills all week is supposed to report zero all week. Reporting both at the same strength would either
    /// cry wolf on every rare-event channel or say nothing about the latency case.</para>
    /// </summary>
    /// <param name="Metric">The built-in channel, or <see cref="MetricIndex.Count"/> for a custom one.</param>
    /// <param name="Name">The custom channel's name, or empty for a built-in.</param>
    /// <param name="Value">The single value the channel has reported throughout.</param>
    /// <param name="Observations">How many observations were behind the judgement.</param>
    public readonly record struct InertChannel(
        MetricIndex Metric,
        string Name,
        double Value,
        int Observations)
    {
        /// <summary>
        /// Whether this is evidence of a defect rather than a question to ask.
        ///
        /// <para>A constant non-zero reading is conclusive: no real measurement of load, latency or memory
        /// returns bit-identical values for hours. A constant zero is not — it is what a correctly bound
        /// rare-event channel looks like on a cluster where the rare event did not happen, and it is also
        /// exactly what a dead binding looks like. The only way to separate those two is to know whether the
        /// event occurred, which this type cannot know and the operator can.</para>
        /// </summary>
        public bool IsConclusive => Value != 0.0;

        /// <summary>The channel's name for a report, whichever kind it is.</summary>
        public string Describe()
        {
            return Name.Length > 0 ? Name : Metric.ToString();
        }
    }
}
