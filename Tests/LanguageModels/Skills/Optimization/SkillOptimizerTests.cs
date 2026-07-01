// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Collections.Generic;
using System.Linq;
using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;
using DevOnBike.Overfit.LanguageModels.Skills.Optimization;

namespace DevOnBike.Overfit.Tests.LanguageModels.Skills.Optimization
{
    /// <summary>
    /// Model-free proof of the SkillOpt loop: an improving editor is accepted only via a strict held-out gain,
    /// and an editor that never improves is always rejected (baseline kept). Fakes stand in for the model so the
    /// control flow is tested deterministically; a real optimizer/editor is validated separately under [LongFact].
    /// </summary>
    public sealed class SkillOptimizerTests
    {
        // Skill passes iff its instructions contain the marker → the editor's job is to introduce it.
        private sealed class MarkerRunner : ISkillRunner
        {
            private readonly bool _good;
            public MarkerRunner(string instructions) => _good = instructions.Contains("MAGIC", StringComparison.Ordinal);
            public SkillRunResult Run(string prompt, bool skillEnabled)
                => new(skillEnabled && _good ? "GOOD" : "BAD", skillEnabled, 1, 1.0);
        }

        private sealed class IsGoodGrader : ISkillGrader
        {
            public string Id => "is_good";
            public GradeCheck Grade(SkillEvalCase testCase, SkillRunResult result)
                => new(Id, result.Output == "GOOD", result.Output);
        }

        // Improving editor: adds the marker once (then proposes nothing).
        private sealed class AddMarkerEditor : ISkillEditor
        {
            public string? Propose(string current, IReadOnlyList<CaseFailure> failures, IReadOnlyList<string> rejected)
                => current.Contains("MAGIC", StringComparison.Ordinal) ? null : current + " MAGIC";
        }

        // Non-improving editor: keeps proposing distinct junk that never introduces the marker.
        private sealed class NoiseEditor : ISkillEditor
        {
            private int _n;
            public string? Propose(string current, IReadOnlyList<CaseFailure> failures, IReadOnlyList<string> rejected)
                => "do it NOISE" + _n++;
        }

        private static CheckRegistry Registry() => new CheckRegistry().Register(new IsGoodGrader());
        private static SkillEvalCase[] Cases(string id) => new[] { new SkillEvalCase(id, "prompt", true, new[] { "is_good" }) };

        [Fact]
        public void ImprovingEdit_IsAccepted_OnStrictHeldOutGain()
        {
            var result = SkillOptimizer.Optimize(
                "do it",
                i => new MarkerRunner(i),
                new AddMarkerEditor(),
                Registry(),
                trainCases: Cases("t"),
                validationCases: Cases("v"),
                rounds: 3);

            Assert.Contains("MAGIC", result.BestInstructions);
            Assert.Equal(1.0, result.BestValidationScore);
            Assert.Contains(result.Steps, s => s.Accepted && s.Round > 0);
        }

        [Fact]
        public void NonImprovingEdits_AreRejected_BaselineKept()
        {
            var result = SkillOptimizer.Optimize(
                "do it",
                i => new MarkerRunner(i),
                new NoiseEditor(),
                Registry(),
                trainCases: Cases("t"),
                validationCases: Cases("v"),
                rounds: 3);

            Assert.Equal("do it", result.BestInstructions);   // never beat the baseline → original kept
            Assert.Equal(0.0, result.BestValidationScore);
            Assert.True(result.Steps.Count(s => !s.Accepted && s.Note.Contains("rejected", StringComparison.Ordinal)) >= 1);
        }
    }
}
