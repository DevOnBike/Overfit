// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One absolute-threshold decision, with its gates kept apart.
    ///
    /// <para>A rule answers a different question from a trend or a peer comparison: not "is this moving" or
    /// "is this one different", but "was it over the line, and for enough of the window". Its two gates are
    /// therefore the threshold and the breach fraction, and a finding needs BOTH — a spike that clears the
    /// threshold for one sample in forty is not what the rule is for.</para>
    ///
    /// <para><see cref="UsableSamples"/> is the third and least visible: <c>MinimumSamples</c> is pinned at
    /// 20 by the config reader, so a window with fewer cannot produce a finding whatever the data does, and
    /// that verdict is <c>WarmingUp</c> rather than <c>Healthy</c>. A replay too short to clear it is silent
    /// by arithmetic, which has been mistaken for a detector failure here more than once.</para>
    /// </summary>
    /// <param name="Signal">Channel name.</param>
    /// <param name="Pod">The pod judged.</param>
    /// <param name="Status">The verdict, or <c>WarmingUp</c> when too few samples existed to reach one.</param>
    /// <param name="Threshold">The line, in the signal's own unit.</param>
    /// <param name="MinBreachFraction">How much of the window had to be over it.</param>
    /// <param name="BreachFraction">
    /// Share of <paramref name="UsableSamples"/> at or above <paramref name="Threshold"/>, on 0…1 — what
    /// <paramref name="MinBreachFraction"/> is compared against. A fraction of the usable samples, not of the
    /// window: a window half full of NaN is judged on the half that survived, so this can read 1.0 on evidence
    /// the reader would not call complete. <paramref name="UsableSamples"/> is what says which it was.
    /// </param>
    /// <param name="BreachedSamples">
    /// How many samples were at or above the threshold — the numerator behind
    /// <paramref name="BreachFraction"/>, carried separately so a fraction near the bar can be read as the
    /// count it came from.
    /// </param>
    /// <param name="UsableSamples">
    /// Observations the verdict rests on, after non-finite values were dropped. Below the rule's
    /// <c>MinimumSamples</c> — 20, as the summary above notes — no verdict is reached at all and
    /// <paramref name="Status"/> is <c>WarmingUp</c>, whatever the other numbers here say.
    /// </param>
    /// <param name="PeakValue">
    /// Highest usable observation, in the signal's own unit, or <see cref="double.NaN"/> when there were none.
    /// Reported because it answers "by how much" — the threshold and the fraction together say a line was
    /// crossed and for how long, and neither says how far past it the signal went.
    /// </param>
    /// <param name="MedianValue">
    /// Median usable observation, in the signal's own unit, or <see cref="double.NaN"/> when there were none.
    /// Read next to <paramref name="PeakValue"/>: a peak far above a median sitting under the threshold is a
    /// spike, and the two close together above it is a signal that has moved.
    /// </param>
    /// <param name="Reason">The detector's own sentence.</param>
    public readonly record struct RuleDecisionTrace(
        string Signal,
        string Pod,
        DetectionStatus Status,
        double Threshold,
        double MinBreachFraction,
        double BreachFraction,
        int BreachedSamples,
        int UsableSamples,
        double PeakValue,
        double MedianValue,
        string Reason);
}
