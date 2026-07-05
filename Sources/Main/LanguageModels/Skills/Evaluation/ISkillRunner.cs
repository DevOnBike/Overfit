// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Runs a prompt against a local model — once with the skill's instructions in context
    /// (<paramref name="skillEnabled"/> = true) and once without — so the evaluator can measure the skill's
    /// <b>lift</b> over the unaided model (the "retire the skill when the bare model matches it" test).
    /// The implementation decides how the skill is injected (system prompt, tool set, …) and controls decoding:
    /// greedy/seeded for a byte-reproducible regression run, or temperature for a capability distribution. This
    /// abstraction keeps the evaluator model-free and unit-testable (fake runners) while the real adapter wraps
    /// <c>OverfitClient</c> / <c>ChatSession</c> / <c>ToolCallConstraint</c>.
    /// </summary>
    public interface ISkillRunner
    {
        SkillRunResult Run(string prompt, bool skillEnabled);
    }
}
