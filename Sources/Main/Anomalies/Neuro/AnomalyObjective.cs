// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Neuro
{
    /// <summary>
    /// What the search is asked to maximise. Every option here is a <b>counting</b> metric over hard
    /// accept/reject decisions — a step function of the threshold, so its gradient is zero almost everywhere and
    /// undefined at the boundary. That is precisely why gradient descent cannot train it and an evolutionary
    /// strategy can: the strategy only ever needs a number, never a derivative.
    /// </summary>
    public enum AnomalyObjective
    {
        /// <summary>Harmonic mean of precision and recall. The default when false alarms and misses hurt about
        /// equally. Robust to the extreme class imbalance typical of anomaly data, where accuracy is useless
        /// (a detector that never fires scores 99.9 %).</summary>
        F1,

        /// <summary>Business cost, minimised (returned negated so bigger is still better). Use when a miss and a
        /// false alarm have genuinely different prices — e.g. a spurious page costs an engineer's hour, a missed
        /// outage costs a customer. This is the objective a real operator argues about in a review, and the one
        /// no differentiable surrogate expresses.</summary>
        Cost,

        /// <summary>Recall subject to a hard cap on false alarms: any candidate exceeding the budget scores 0.
        /// Encodes "catch as much as possible, but never page us more than N times a day" — a constraint, not a
        /// trade-off, and therefore not expressible as a weighted loss at all.</summary>
        RecallAtFalseAlarmBudget,
    }
}
