// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Maps check ids to <see cref="ISkillGrader"/>s and dispatches, per case, only the checks that case
    /// declares in <see cref="SkillEvalCase.ExpectedChecks"/>. This id-keyed dispatch is what lets the rule set
    /// grow without a combinatorial per-case explosion. An unknown check id fails loudly (a typo is a test bug,
    /// not a silent pass).
    /// </summary>
    public sealed class CheckRegistry
    {
        private readonly Dictionary<string, ISkillGrader> _graders = new(StringComparer.Ordinal);

        public CheckRegistry Register(ISkillGrader grader)
        {
            ArgumentNullException.ThrowIfNull(grader);
            
            _graders[grader.Id] = grader;
            
            return this;
        }

        /// <summary>Runs the checks named by <paramref name="testCase"/> against <paramref name="result"/>.</summary>
        public IReadOnlyList<GradeCheck> Grade(SkillEvalCase testCase, SkillRunResult result)
        {
            ArgumentNullException.ThrowIfNull(testCase);

            var checks = new List<GradeCheck>(testCase.ExpectedChecks.Count);
            
            foreach (var id in testCase.ExpectedChecks)
            {
                if (_graders.TryGetValue(id, out var grader))
                {
                    checks.Add(grader.Grade(testCase, result));
                }
                else
                {
                    checks.Add(new GradeCheck(id, false, "no grader registered for this check id"));
                }
            }

            return checks;
        }
    }
}
