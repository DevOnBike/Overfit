// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Neuro
{
    /// <summary>
    /// Which objective the evolutionary search maximises, and its knobs. A readonly struct passed by <c>in</c> —
    /// the fitness function runs population × samples × generations times, so this must not allocate.
    /// </summary>
    public readonly struct AnomalyFitnessOptions
    {
        /// <summary>Defaults to <see cref="AnomalyObjective.F1"/> with symmetric costs.</summary>
        public AnomalyFitnessOptions()
        {
        }

        /// <summary>What to maximise. Default <see cref="AnomalyObjective.F1"/>.</summary>
        public AnomalyObjective Objective { get; init; } = AnomalyObjective.F1;

        /// <summary>Price of a MISSED anomaly. Used by <see cref="AnomalyObjective.Cost"/>. Typically much
        /// larger than <see cref="FalseAlarmCost"/> — that asymmetry is the reason to use the Cost objective.</summary>
        public float MissCost { get; init; } = 10f;

        /// <summary>Price of a FALSE ALARM. Used by <see cref="AnomalyObjective.Cost"/>.</summary>
        public float FalseAlarmCost { get; init; } = 1f;

        /// <summary>Maximum false alarms tolerated by <see cref="AnomalyObjective.RecallAtFalseAlarmBudget"/>;
        /// exceeding it scores 0 (a hard constraint, not a penalty term).</summary>
        public int FalseAlarmBudget { get; init; } = 5;
    }
}
