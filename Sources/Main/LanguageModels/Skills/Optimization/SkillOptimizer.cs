// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;

namespace DevOnBike.Overfit.LanguageModels.Skills.Optimization
{
    /// <summary>
    /// Weight-free, text-space skill self-improvement (SkillOpt): iteratively edit a skill's instruction text and
    /// keep an edit ONLY when it strictly raises a held-out validation score. Each round scores the current skill
    /// on the TRAIN split, hands its failures to an <see cref="ISkillEditor"/> (which proposes one revision), then
    /// scores that candidate on the VALIDATION split — the selection gate. Rejected candidates go into a buffer
    /// fed back to the editor so it doesn't repeat them.
    ///
    /// <para>This is the "learn without touching weights" loop, and it rides entirely on Overfit primitives:
    /// local generation (the editor + the runner), <see cref="SkillEvaluator"/> for scoring, and a schema-locked
    /// editor — all offline and reproducible. The held-out gate is exactly the guard against overfitting the
    /// prompt to the train cases. Model-free by construction (the runner factory + editor are injected), so the
    /// loop logic is unit-testable without a model.</para>
    /// </summary>
    public static class SkillOptimizer
    {
        /// <param name="initialInstructions">The starting skill instruction text.</param>
        /// <param name="runnerFactory">Builds a runner for a given candidate instruction text (real impl:
        /// <c>i => new OverfitSkillRunner(client, name, desc, i)</c>).</param>
        /// <param name="editor">Proposes one bounded revision per round.</param>
        /// <param name="registry">The deterministic (and/or rubric) checks used to score.</param>
        /// <param name="trainCases">Cases used to surface failures for the editor.</param>
        /// <param name="validationCases">Held-out cases; only a strict improvement here is accepted.</param>
        /// <param name="rounds">Max optimization rounds.</param>
        public static SkillOptResult Optimize(
            string initialInstructions,
            Func<string, ISkillRunner> runnerFactory,
            ISkillEditor editor,
            CheckRegistry registry,
            IReadOnlyList<SkillEvalCase> trainCases,
            IReadOnlyList<SkillEvalCase> validationCases,
            int rounds)
        {
            ArgumentNullException.ThrowIfNull(initialInstructions);
            ArgumentNullException.ThrowIfNull(runnerFactory);
            ArgumentNullException.ThrowIfNull(editor);
            ArgumentNullException.ThrowIfNull(registry);
            ArgumentNullException.ThrowIfNull(trainCases);
            ArgumentNullException.ThrowIfNull(validationCases);

            var current = initialInstructions;
            var currentVal = ScorePassRate(current, runnerFactory, registry, validationCases);

            var steps = new List<SkillOptResult.Step> { new(0, current, currentVal, true, "baseline") };
            var rejected = new List<string>();

            for (var round = 1; round <= rounds; round++)
            {
                var failures = CollectTrainFailures(current, runnerFactory, registry, trainCases);
                if (failures.Count == 0)
                {
                    steps.Add(new SkillOptResult.Step(round, current, currentVal, false, "no train failures — stop"));
                    break;
                }

                var candidate = editor.Propose(current, failures, rejected);
                if (string.IsNullOrWhiteSpace(candidate) || string.Equals(candidate, current, StringComparison.Ordinal))
                {
                    steps.Add(new SkillOptResult.Step(round, current, currentVal, false, "editor proposed no change"));
                    continue;
                }

                var candidateVal = ScorePassRate(candidate!, runnerFactory, registry, validationCases);

                // Selection gate: keep the edit ONLY on a strict held-out improvement (guards against overfitting
                // the prompt to the train split).
                if (candidateVal > currentVal)
                {
                    current = candidate!;
                    currentVal = candidateVal;
                    steps.Add(new SkillOptResult.Step(round, candidate!, candidateVal, true, "accepted (val improved)"));
                }
                else
                {
                    rejected.Add(candidate!);
                    steps.Add(new SkillOptResult.Step(round, candidate!, candidateVal, false, "rejected (no val improvement)"));
                }
            }

            return new SkillOptResult(current, currentVal, steps);
        }

        private static double ScorePassRate(
            string instructions,
            Func<string, ISkillRunner> runnerFactory,
            CheckRegistry registry,
            IReadOnlyList<SkillEvalCase> cases)
        {
            var report = SkillEvaluator.Evaluate(cases, runnerFactory(instructions), registry);
            return report.PassRateOn;
        }

        private static List<CaseFailure> CollectTrainFailures(
            string instructions,
            Func<string, ISkillRunner> runnerFactory,
            CheckRegistry registry,
            IReadOnlyList<SkillEvalCase> trainCases)
        {
            var report = SkillEvaluator.Evaluate(trainCases, runnerFactory(instructions), registry);
            var failures = new List<CaseFailure>();
            for (var i = 0; i < report.Cases.Count; i++)
            {
                var c = report.Cases[i];
                if (!c.OnPass)
                {
                    failures.Add(new CaseFailure(c.Case.Prompt, c.OnResult.Output));
                }
            }
            return failures;
        }
    }
}
