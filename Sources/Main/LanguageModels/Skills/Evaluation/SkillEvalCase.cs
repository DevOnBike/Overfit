// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// One eval case for an agent skill/prompt: a user <paramref name="Prompt"/>, whether the skill is expected
    /// to activate (<paramref name="ShouldTrigger"/> — include negative controls so over-triggering is caught,
    /// not just failures), and the ids of the named checks that apply to this case (dispatched from a
    /// <see cref="CheckRegistry"/>, so rules scale without a combinatorial per-case explosion).
    /// </summary>
    public sealed record SkillEvalCase(
        string Id,
        string Prompt,
        bool ShouldTrigger,
        IReadOnlyList<string> ExpectedChecks);
}
