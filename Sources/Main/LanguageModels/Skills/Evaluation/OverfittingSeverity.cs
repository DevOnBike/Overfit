// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// How badly an eval definition tests the skill's WORDING rather than its OUTCOME
    /// (see <see cref="OverfittingJudge"/>). Banded from the 0..1 score: Low &lt; 0.20 ≤ Moderate &lt; 0.50 ≤ High.
    /// </summary>
    public enum OverfittingSeverity
    {
        /// <summary>The judge classified NOTHING (unusable/truncated reply), so the eval was never assessed.
        /// Distinct from <see cref="Low"/> on purpose: reporting a failed judge as "healthy" is the exact lie
        /// this check exists to prevent. A 0.0 score alongside this means "unknown", not "clean".</summary>
        Unknown,

        /// <summary>Score &lt; 0.20 — the eval mostly tests outcomes. Healthy.</summary>
        Low,

        /// <summary>0.20 ≤ score &lt; 0.50 — some criteria reward the skill's method or vocabulary. Worth a review.</summary>
        Moderate,

        /// <summary>Score ≥ 0.50 — the eval largely rewards parroting the skill. A passing score here proves little.</summary>
        High,
    }
}
