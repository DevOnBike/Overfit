// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Statistics;

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// One trend decision, with the gates that produced it kept apart.
    ///
    /// <para><b>Why a separate record from <see cref="PeerDecisionTrace"/>.</b> That one is shaped by the
    /// peer comparison — relative gap, absolute gap, effect size — and a trend has none of those. It has a
    /// slope, a rank correlation, an autocorrelation and a floor on the change across the window, and
    /// forcing them into peer-shaped fields would mean a reader could not tell which number meant what.</para>
    ///
    /// <para><b><see cref="WarmingUp"/> is the field this was worth building for.</b> A pod inside the
    /// warm-up grace is skipped with a bare <c>continue</c> and, before this, left no evidence at all — so
    /// "no finding" and "not even looked at" were the same silence. During a rollout that is every pod.</para>
    /// </summary>
    /// <param name="Signal">Channel name, built-in or custom.</param>
    /// <param name="Pod">The pod judged.</param>
    /// <param name="Status">The verdict, or <c>InsufficientData</c> when none was reached.</param>
    /// <param name="WarmingUp">The pod was inside the warm-up grace and no test ran.</param>
    /// <param name="FloorOverWindow">The change the signal had to clear, in its own unit.</param>
    /// <param name="HasExpectation">A seasonal expectation was subtracted before the fit.</param>
    /// <param name="Reason">The detector's own sentence, which is what an operator reads first.</param>
    public readonly record struct TrendDecisionTrace(
        string Signal,
        string Pod,
        DetectionStatus Status,
        bool WarmingUp,
        double FloorOverWindow,
        double SlopePerSecond,
        double KendallTau,
        double PValue,
        double Autocorrelation,
        int SampleCount,
        bool HasExpectation,
        string Reason);
}
