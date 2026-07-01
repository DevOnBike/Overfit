// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Optimization
{
    /// <summary>
    /// Outcome of a <see cref="SkillOptimizer"/> run: the best skill instructions found, its held-out
    /// validation score, and the full accept/reject trace (so an improvement can be audited — every accepted
    /// step strictly raised the validation score, every rejected one didn't).
    /// </summary>
    public sealed class SkillOptResult
    {
        /// <summary>One optimization round: the candidate instructions, its validation score, whether it was
        /// accepted (strict improvement), and a short note.</summary>
        public sealed record Step(int Round, string Instructions, double ValidationScore, bool Accepted, string Note);

        public string BestInstructions { get; }
        public double BestValidationScore { get; }
        public IReadOnlyList<Step> Steps { get; }

        public SkillOptResult(string bestInstructions, double bestValidationScore, IReadOnlyList<Step> steps)
        {
            BestInstructions = bestInstructions;
            BestValidationScore = bestValidationScore;
            Steps = steps;
        }
    }
}
