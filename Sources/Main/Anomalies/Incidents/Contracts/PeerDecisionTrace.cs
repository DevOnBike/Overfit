// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Incidents.Contracts
{
    /// <summary>
    /// The numbers that decided one peer comparison, for one member.
    ///
    /// <para><b>Every gate is reported separately because "no finding" has five different causes</b> and they
    /// call for opposite fixes. Too few usable samples is a data problem; an effect size under the bar is the
    /// masking bound biting; a relative gap under the bar is a threshold question; a group-level
    /// <c>Inconclusive</c> means members crossed in both directions and no culprit was named. A single
    /// boolean cannot tell those apart, and guessing between them has cost this project repeatedly.</para>
    /// </summary>
    /// <param name="Signal">Metric compared.</param>
    /// <param name="Status">The group's verdict.</param>
    /// <param name="High">Members found above the rest.</param>
    /// <param name="Low">Members found below.</param>
    /// <param name="Pod">The member this row is about.</param>
    /// <param name="IsOutlier">Whether it was reported.</param>
    /// <param name="RelativeGap">Its distance from the other members' median, as a fraction of theirs.</param>
    /// <param name="AbsoluteGap">The same distance in the metric's own units.</param>
    /// <param name="EffectSize">Cliff's delta — how little the distributions overlap.</param>
    /// <param name="PValue">Significance before the group's Bonferroni correction is applied.</param>
    /// <param name="UsableSamples">Finite observations it contributed.</param>
    /// <param name="ExcludedPeers">
    /// Members dropped from this comparison for contributing too few usable samples. A sixth cause of "no
    /// finding", and the one that hides behind the other five: the comparison ran, said something, and said it
    /// about a smaller group than the reader assumes.
    /// </param>
    public readonly record struct PeerDecisionTrace(
        string Signal,
        DetectionStatus Status,
        int High,
        int Low,
        string Pod,
        bool IsOutlier,
        double RelativeGap,
        double AbsoluteGap,
        double EffectSize,
        double PValue,
        int UsableSamples,
        int ExcludedPeers = 0);
}
