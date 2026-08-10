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
