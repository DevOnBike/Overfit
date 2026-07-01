// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>Verdict of a single named check against a run — the atomic unit both the deterministic
    /// graders and the (schema-locked) rubric grader produce.</summary>
    public sealed record GradeCheck(string Id, bool Pass, string Notes);
}
