// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// The verdict of an <see cref="OverfittingJudge"/> run: a 0..1 <paramref name="Score"/> (higher = more
    /// overfitted) with its <paramref name="Severity"/> band, plus the per-element classifications the score was
    /// computed from. <b>Informational, never a gate</b> — the thresholds are uncalibrated, so this reports a
    /// warning about the EVAL's design; it does not fail a skill.
    /// </summary>
    public sealed record OverfittingResult(
        double Score,
        OverfittingSeverity Severity,
        IReadOnlyList<OverfittingResult.RubricAssessment> RubricAssessments,
        IReadOnlyList<OverfittingResult.AssertionAssessment> AssertionAssessments,
        string OverallReasoning)
    {
        /// <summary>
        /// One rubric criterion, classified. <paramref name="Classification"/> is one of:
        /// <c>outcome</c> (tests WHAT was achieved — healthy, weight 0.0),
        /// <c>technique</c> (tests a method/diagnostic step the skill teaches — weight 0.5),
        /// <c>vocabulary</c> (tests the skill's exact wording/labels — weight 1.0).
        /// <paramref name="Confidence"/> is 0..1 and scales the weight.
        /// </summary>
        public sealed record RubricAssessment(
            string Criterion,
            string Classification,
            double Confidence,
            string Reasoning);

        /// <summary>
        /// One declared check (deterministic grader), classified. <paramref name="Classification"/> is one of:
        /// <c>broad</c> (a correct alternative approach still passes — weight 0.0) or
        /// <c>narrow</c> (only the skill's specific pattern passes — weight 1.0).
        /// </summary>
        public sealed record AssertionAssessment(
            string Check,
            string Classification,
            double Confidence,
            string Reasoning);
    }
}
