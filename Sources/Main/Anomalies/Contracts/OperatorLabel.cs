// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One judgement an operator made about one incident.
    ///
    /// <para><b>The magnitude is the part that does the work.</b> A label saying "incident 42 was real" cannot
    /// be checked against a threshold; a label saying "a peer gap of 6.3 MB on <c>MemoryWorkingSetBytes</c>
    /// was real" can, and that is what stops a later calibration from proposing a floor of 8 MB and quietly
    /// removing the guard's ability to see it again. It is carried in the <b>signal's own units</b>, the same
    /// units the floors are compared in.</para>
    ///
    /// <para><b>Why the incident id and not the pod.</b> The pod is gone by the next deploy; the identity a
    /// human acted on is the incident's. That identity is only trustworthy because
    /// <c>IncidentTracker.Restore</c> stopped recycling identifiers on 2026-08-02 — before that fix an
    /// acknowledgement could have landed on an unrelated incident, which is a worse failure than having no
    /// acknowledgement at all.</para>
    /// </summary>
    /// <param name="IncidentId">The incident the operator acted on.</param>
    /// <param name="Signal">Metric name the judgement is about.</param>
    /// <param name="Kind">Noise or real.</param>
    /// <param name="Magnitude">
    /// How large the finding was, in the signal's own units, or <see cref="double.NaN"/> when the detector
    /// that produced it reports no absolute size. A label with no magnitude still records the operator's
    /// opinion and simply cannot constrain a threshold.
    /// </param>
    /// <param name="At">When the label was recorded.</param>
    /// <param name="Reason">What the operator typed, or empty.</param>
    public readonly record struct OperatorLabel(
        long IncidentId,
        string Signal,
        OperatorLabelKind Kind,
        double Magnitude,
        DateTimeOffset At,
        string Reason)
    {
        /// <summary>Whether this label can constrain a proposed floor.</summary>
        public bool ConstrainsFloors => Kind == OperatorLabelKind.Real && double.IsFinite(Magnitude)
                                        && Magnitude > 0.0;
    }
}
