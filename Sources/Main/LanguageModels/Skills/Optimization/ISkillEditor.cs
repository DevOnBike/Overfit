// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Optimization
{
    /// <summary>
    /// Proposes ONE bounded revision of a skill's instruction text, given the cases it currently fails and the
    /// revisions already tried-and-rejected (so it doesn't loop). This is the "optimizer" of the SkillOpt loop;
    /// abstracting it keeps <see cref="SkillOptimizer"/> model-free and unit-testable, while the real
    /// implementation (<see cref="OverfitSkillEditor"/>) is a local model emitting a schema-locked edit. Return
    /// <c>null</c> (or the unchanged input) to signal "no edit this round".
    /// </summary>
    public interface ISkillEditor
    {
        string? Propose(
            string currentInstructions,
            IReadOnlyList<CaseFailure> failures,
            IReadOnlyList<string> rejectedRevisions);
    }
}
