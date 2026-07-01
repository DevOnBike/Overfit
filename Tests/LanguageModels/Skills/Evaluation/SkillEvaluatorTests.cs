// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;

namespace DevOnBike.Overfit.Tests.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Model-free proof of the skill-eval core: the ON/OFF baseline arm, the id-keyed check dispatch, and the
    /// aggregates (pass rates, lift, trigger accuracy). Uses fake runners/graders so it's fast and deterministic
    /// (a real local-model runner is validated separately under [LongFact]).
    /// </summary>
    public sealed class SkillEvaluatorTests
    {
        // A grader that passes iff the output contains a marker token — stands in for any deterministic check.
        private sealed class ContainsGrader : ISkillGrader
        {
            private readonly string _needle;
            
            public ContainsGrader(string id, string needle)
            {
                Id = id;
                _needle = needle;
            }
            public string Id { get; }
            public GradeCheck Grade(SkillEvalCase testCase, SkillRunResult result)
                => new(Id, result.Output.Contains(_needle, StringComparison.Ordinal), _needle);
        }

        // Skill ON injects the marker (so the check passes); OFF does not (baseline fails) → lift = 1.
        private sealed class MarkerRunner : ISkillRunner
        {
            public SkillRunResult Run(string prompt, bool skillEnabled)
                => new(skillEnabled ? "MARKER " + prompt : prompt, Triggered: skillEnabled, Tokens: 5, ElapsedMs: 1.0);
        }

        private static SkillEvalCase Case(string id, string prompt, params string[] checks)
            => new(id, prompt, ShouldTrigger: true, checks);

        [Fact]
        public void Evaluate_ComputesLift_PassRates_AndTriggerAccuracy()
        {
            var cases = new List<SkillEvalCase>
            {
                Case("a", "do the thing", "has_marker"),
                Case("b", "do another thing", "has_marker"),
            };
            var registry = new CheckRegistry().Register(new ContainsGrader("has_marker", "MARKER"));

            var report = SkillEvaluator.Evaluate(cases, new MarkerRunner(), registry);

            Assert.Equal(1.0, report.PassRateOn);   // skill ON injects the marker → all pass
            Assert.Equal(0.0, report.PassRateOff);  // unaided baseline never has it
            Assert.Equal(1.0, report.Lift);         // the skill is fully responsible for the pass
            Assert.Equal(1.0, report.TriggerAccuracy);
            Assert.Equal(2, report.Cases.Count);
        }

        [Fact]
        public void UnknownCheckId_FailsLoudly_NotSilentPass()
        {
            var cases = new List<SkillEvalCase> { Case("a", "x", "typo_check") };
            var registry = new CheckRegistry(); // nothing registered

            var report = SkillEvaluator.Evaluate(cases, new MarkerRunner(), registry);

            Assert.False(report.Cases[0].OnPass);
            Assert.False(report.Cases[0].OnChecks[0].Pass);
            Assert.Contains("no grader", report.Cases[0].OnChecks[0].Notes, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void NegativeControl_WrongTrigger_LowersTriggerAccuracy()
        {
            // Case says the skill should NOT trigger, but the MarkerRunner always triggers when enabled → miss.
            var cases = new List<SkillEvalCase>
            {
                new("neg", "unrelated request", ShouldTrigger: false, new[] { "has_marker" }),
            };
            var registry = new CheckRegistry().Register(new ContainsGrader("has_marker", "MARKER"));

            var report = SkillEvaluator.Evaluate(cases, new MarkerRunner(), registry);

            Assert.Equal(0.0, report.TriggerAccuracy); // over-triggered on a negative control
        }
    }
}
