// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// A named, deterministic check ("did it do the basics?") — a model-free predicate over a run's output
    /// (string/regex match, does the JSON parse, was a required tool used, is the token budget respected). Keep
    /// these fast and deterministic so they run in CI; the subjective "did it follow conventions?" dimension is
    /// the separate schema-locked rubric grader. Register instances in a <see cref="CheckRegistry"/> under
    /// <see cref="Id"/>; a <see cref="SkillEvalCase"/> lists the ids that apply to it.
    /// </summary>
    public interface ISkillGrader
    {
        /// <summary>The check id — matched against <see cref="SkillEvalCase.ExpectedChecks"/>.</summary>
        string Id { get; }

        GradeCheck Grade(SkillEvalCase testCase, SkillRunResult result);
    }
}
