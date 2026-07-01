// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Runs an agent-skill eval entirely against a local model: every case is executed through the
    /// <see cref="ISkillRunner"/> twice — skill ON and skill OFF — graded by the <see cref="CheckRegistry"/>,
    /// and summarised as pass rates + the ON-vs-OFF <see cref="SkillEvalReport.Lift"/> + trigger accuracy.
    ///
    /// <para>The ON/OFF baseline arm and the (optional) byte-reproducible greedy/seeded runner are the point:
    /// a cloud grader pays per trial and is inherently stochastic, so nobody runs the unaided baseline; locally
    /// it's free, so the lift is measured by default and a regression run is exactly reproducible.</para>
    ///
    /// <para>This is deterministic single-run mode (one ON + one OFF per case). Distribution mode — N trials
    /// under temperature for capability evals — and the schema-locked rubric grader (a local judge via
    /// <c>JsonSchemaConstraint</c>) layer on top of this same shape.</para>
    /// </summary>
    public static class SkillEvaluator
    {
        /// <summary>A case passes when it has at least one check and all its checks pass (an empty check list
        /// is not a meaningful pass).</summary>
        private static bool AllPass(IReadOnlyList<GradeCheck> checks)
        {
            if (checks.Count == 0)
            {
                return false;
            }
            
            for (var i = 0; i < checks.Count; i++)
            {
                if (!checks[i].Pass)
                {
                    return false;
                }
            }
            
            return true;
        }

        public static SkillEvalReport Evaluate(IReadOnlyList<SkillEvalCase> cases, ISkillRunner runner, CheckRegistry registry)
        {
            ArgumentNullException.ThrowIfNull(cases);
            ArgumentNullException.ThrowIfNull(runner);
            ArgumentNullException.ThrowIfNull(registry);

            var results = new List<SkillEvalReport.CaseResult>(cases.Count);
            
            foreach (var c in cases)
            {
                var on = runner.Run(c.Prompt, skillEnabled: true);
                var off = runner.Run(c.Prompt, skillEnabled: false);

                var onChecks = registry.Grade(c, on);
                var offChecks = registry.Grade(c, off);

                // Trigger is graded only when the runner reports a decision; a runner that doesn't model
                // triggering leaves it null and is treated as correct (not penalised).
                var triggerCorrect = !on.Triggered.HasValue || on.Triggered.Value == c.ShouldTrigger;

                results.Add(new SkillEvalReport.CaseResult(
                    c, on, onChecks, AllPass(onChecks), off, offChecks, AllPass(offChecks), triggerCorrect));
            }

            return new SkillEvalReport(results);
        }
    }
}
