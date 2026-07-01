// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// The captured result of running one prompt against the model. <paramref name="Triggered"/> is the
    /// (deterministic) skill/tool-selection decision when the runner measures it (e.g. via a
    /// <c>ToolCallConstraint</c> name-enum), or <c>null</c> when the run doesn't model triggering.
    /// <paramref name="Tokens"/>/<paramref name="ElapsedMs"/> feed the efficiency dimension.
    /// </summary>
    public sealed record SkillRunResult(
        string Output,
        bool? Triggered,
        int Tokens,
        double ElapsedMs);
}
