// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Optimization
{
    /// <summary>A training case the current skill got wrong — the prompt and what the skill actually produced.
    /// The optimizer reasons over these to propose a targeted edit.</summary>
    public sealed record CaseFailure(string Prompt, string Output);
}
