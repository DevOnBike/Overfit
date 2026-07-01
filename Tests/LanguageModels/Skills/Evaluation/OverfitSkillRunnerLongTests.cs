// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// End-to-end skill eval on a REAL local model: <see cref="OverfitSkillRunner"/> runs a "be concise" skill
    /// ON vs OFF, a deterministic length grader + the schema-locked <see cref="RubricGrader"/> score it, and
    /// <see cref="SkillEvaluator"/> reports pass rates + lift + trigger accuracy. <see cref="LongFact"/> — needs
    /// <c>OVERFIT_QWEN3B_DIR</c> with the Q4_K_M GGUF. Asserts the mechanics (not a specific lift number, which
    /// is model-dependent); the number is logged for manual inspection.
    /// </summary>
    public sealed class OverfitSkillRunnerLongTests
    {
        private readonly ITestOutputHelper _out;
        public OverfitSkillRunnerLongTests(ITestOutputHelper output) => _out = output;

        // Passes when the answer is short (the skill's whole point) — a model-free deterministic check.
        private sealed class ShortAnswerGrader : ISkillGrader
        {
            public string Id => "short_answer";
            public GradeCheck Grade(SkillEvalCase testCase, SkillRunResult result)
            {
                var words = result.Output.Split(
                    new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries).Length;
                return new GradeCheck(Id, words > 0 && words <= 6, $"{words} words");
            }
        }

        [LongFact]
        public void ConciseSkill_RunsEndToEnd_ReportsLiftTriggerAndRubric()
        {
            var gguf = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            using var client = OverfitClient.LoadGguf(gguf, maxContextLength: 2048, maxNewTokens: 128);

            var runner = new OverfitSkillRunner(
                client,
                skillName: "concise_answer",
                skillDescription: "Answer a short factual question with just the answer and nothing else.",
                skillInstructions: "Reply with ONLY the direct answer in as few words as possible. No sentence, no explanation.");

            var registry = new CheckRegistry()
                .Register(new ShortAnswerGrader())
                .Register(new RubricGrader(client,
                    rubric: "Pass only if the answer is correct AND at most a few words (no full sentence).",
                    id: "concise_and_correct"));

            var cases = new List<SkillEvalCase>
            {
                new("fr", "What is the capital of France?", ShouldTrigger: true, new[] { "short_answer", "concise_and_correct" }),
                new("jp", "What is the capital of Japan?", ShouldTrigger: true, new[] { "short_answer", "concise_and_correct" }),
            };

            var report = SkillEvaluator.Evaluate(cases, runner, registry);

            foreach (var c in report.Cases)
            {
                _out.WriteLine($"[{c.Case.Id}] trigger={c.TriggerCorrect} ON(pass={c.OnPass}): \"{c.OnResult.Output.Trim()}\"");
                _out.WriteLine($"       OFF(pass={c.OffPass}): \"{c.OffResult.Output.Trim()}\"");
            }
            _out.WriteLine($"PassRate ON={report.PassRateOn:P0} OFF={report.PassRateOff:P0} Lift={report.Lift:P0} Trigger={report.TriggerAccuracy:P0}");

            Assert.Equal(2, report.Cases.Count);
            Assert.InRange(report.PassRateOn, 0.0, 1.0);
            Assert.InRange(report.PassRateOff, 0.0, 1.0);
            Assert.InRange(report.TriggerAccuracy, 0.0, 1.0);
            foreach (var c in report.Cases)
            {
                Assert.False(string.IsNullOrWhiteSpace(c.OnResult.Output)); // the model actually produced an answer
                Assert.True(c.OnResult.Triggered.HasValue);                 // trigger was measured (ToolCallConstraint)
            }
        }
    }
}
