// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;
using DevOnBike.Overfit.LanguageModels.Skills.Optimization;

namespace DevOnBike.Overfit.Tests.LanguageModels.Skills.Optimization
{
    /// <summary>
    /// Guards the failure-reason channel. The reason is the ONLY gradient information the editor receives:
    /// measured on Qwen2.5-3B, the same failing case produced "Answer in the format 'The [X] is [Y]'" without a
    /// reason (the wrong direction — it codified the failing answer) and "Answer the question in at most 4 words"
    /// with one. If a refactor stops populating it the loop silently degrades to random text search, with every
    /// other test still green — hence a test.
    /// </summary>
    public sealed class CaseFailureReasonTests
    {
        private sealed class AlwaysFailsGrader : ISkillGrader
        {
            public string Id => "too_long";

            public GradeCheck Grade(SkillEvalCase testCase, SkillRunResult result)
                => new(Id, false, "answer must be at most 4 words");
        }

        private sealed class FixedRunner : ISkillRunner
        {
            public SkillRunResult Run(string prompt, bool skillEnabled)
                => new("The capital of France is Paris.", null, 8, 1.0);
        }

        /// <summary>Captures what the optimizer hands the editor.</summary>
        private sealed class CapturingEditor : ISkillEditor
        {
            public List<CaseFailure> Seen { get; } = [];

            public string? Propose(
                string currentInstructions,
                IReadOnlyList<CaseFailure> failures,
                IReadOnlyList<string> rejectedRevisions)
            {
                Seen.AddRange(failures);
                return null;   // no edit — we only care about what was passed in
            }
        }

        [Fact]
        public void Optimizer_PassesTheFailedCheckAndItsNote_ToTheEditor()
        {
            var editor = new CapturingEditor();
            var registry = new CheckRegistry().Register(new AlwaysFailsGrader());
            var cases = new List<SkillEvalCase> { new("c1", "What is the capital of France?", true, ["too_long"]) };

            SkillOptimizer.Optimize(
                "Answer the question.", _ => new FixedRunner(), editor, registry, cases, cases, rounds: 1);

            var failure = Assert.Single(editor.Seen);
            Assert.Contains("too_long", failure.Reason);                        // which check
            Assert.Contains("at most 4 words", failure.Reason);                 // and its note — the actionable bit
            Assert.Equal("The capital of France is Paris.", failure.Output);
        }

        [Fact]
        public void Reason_IsOptional_SoExistingTwoArgumentCallersStillCompile()
        {
            // Public API on a shipped package: adding a required positional field would be a breaking change.
            var legacy = new CaseFailure("prompt", "output");

            Assert.Equal(string.Empty, legacy.Reason);
        }
    }
}
