// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;

namespace DevOnBike.Overfit.Tests.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Pins <see cref="LiftSignificance"/>. The exact McNemar p-values are hand-computable — for b discordant
    /// pairs all favouring the skill, p = 2·(1/2)^b — so these are real oracles, not golden-file snapshots.
    /// </summary>
    public sealed class LiftSignificanceTests
    {
        [Fact]
        public void NoDiscordantPairs_IsNoEvidence_NotSignificant()
        {
            // Every case agreed (both arms pass, or both fail) → the arms are indistinguishable.
            var s = LiftSignificance.Compute(helped: 0, hurt: 0, total: 20);

            Assert.Equal(0.0, s.Lift, 6);
            Assert.Equal(1.0, s.PValue, 6);
            Assert.False(s.IsSignificant);
            Assert.Equal(0.0, s.LowerBound, 6);
            Assert.Equal(0.0, s.UpperBound, 6);
        }

        [Fact]
        public void ZeroCases_DoesNotDivideByZero()
        {
            var s = LiftSignificance.Compute(helped: 0, hurt: 0, total: 0);

            Assert.Equal(0.0, s.Lift, 6);
            Assert.Equal(1.0, s.PValue, 6);
            Assert.False(s.IsSignificant);
        }

        // p = 2 * P(X >= b) for X~Bin(b, 0.5) with hurt=0  =>  p = 2 * 0.5^b.
        // The punchline: FIVE straight wins is still not significant at 0.05 — you need six.
        [Theory]
        [InlineData(1, 1.0)]
        [InlineData(2, 0.5)]
        [InlineData(3, 0.25)]
        [InlineData(4, 0.125)]
        [InlineData(5, 0.0625)]
        [InlineData(6, 0.03125)]
        public void AllDiscordantFavourSkill_MatchesExactBinomial(int helped, double expectedP)
        {
            var s = LiftSignificance.Compute(helped, hurt: 0, total: 20);

            Assert.Equal(expectedP, s.PValue, 6);
            Assert.Equal(expectedP < 0.05, s.IsSignificant);
        }

        [Fact]
        public void FiveOfFiveWins_LooksHuge_ButIsNotSignificant()
        {
            // The whole reason this type exists: +100% lift on 5 cases, p = 0.0625 → cannot claim it.
            var s = LiftSignificance.Compute(helped: 5, hurt: 0, total: 5);

            Assert.Equal(1.0, s.Lift, 6);
            Assert.False(s.IsSignificant);
        }

        [Fact]
        public void MixedDiscordant_MatchesExactBinomial()
        {
            // b=8, c=2, n=10 discordant: P(X>=8) = (45+10+1)/1024 = 0.0546875; p = 0.109375 → not significant.
            var s = LiftSignificance.Compute(helped: 8, hurt: 2, total: 20);

            Assert.Equal(0.109375, s.PValue, 6);
            Assert.False(s.IsSignificant);
            Assert.Equal(0.3, s.Lift, 6);   // (8-2)/20
        }

        [Fact]
        public void SkillThatHurts_IsSignificantlyNegative()
        {
            var s = LiftSignificance.Compute(helped: 0, hurt: 6, total: 20);

            Assert.True(s.Lift < 0);
            Assert.Equal(0.03125, s.PValue, 6);
            Assert.True(s.IsSignificant);       // significant HARM — the sign lives in Lift, not the p-value
            Assert.True(s.UpperBound < 0);
        }

        [Fact]
        public void PValue_IsSymmetric_UnderSwappingHelpedAndHurt()
        {
            var win = LiftSignificance.Compute(helped: 7, hurt: 2, total: 20);
            var loss = LiftSignificance.Compute(helped: 2, hurt: 7, total: 20);

            Assert.Equal(win.PValue, loss.PValue, 9);
            Assert.Equal(win.Lift, -loss.Lift, 9);
        }

        [Fact]
        public void Interval_BracketsTheLift_AndStaysInRange()
        {
            var s = LiftSignificance.Compute(helped: 9, hurt: 1, total: 12);

            Assert.True(s.LowerBound <= s.Lift);
            Assert.True(s.Lift <= s.UpperBound);
            Assert.InRange(s.LowerBound, -1.0, 1.0);
            Assert.InRange(s.UpperBound, -1.0, 1.0);
        }

        [Fact]
        public void Report_Significance_AgreesWithItsOwnLift()
        {
            // The identity that makes McNemar the right test here: PassRateOn - PassRateOff == (helped-hurt)/n.
            // 6 helped, 1 hurt, 3 concordant passes → on=9, off=4 of 10.
            var cases = new List<SkillEvalReport.CaseResult>();
            for (var i = 0; i < 6; i++)
            {
                cases.Add(Case($"helped{i}", onPass: true, offPass: false));
            }
            cases.Add(Case("hurt0", onPass: false, offPass: true));
            for (var i = 0; i < 3; i++)
            {
                cases.Add(Case($"both{i}", onPass: true, offPass: true));
            }

            var report = new SkillEvalReport(cases);

            Assert.Equal(0.9, report.PassRateOn, 6);
            Assert.Equal(0.4, report.PassRateOff, 6);
            Assert.NotNull(report.Significance);
            Assert.Equal(report.Lift, report.Significance!.Lift, 9);   // 0.5 == (6-1)/10
            Assert.Equal(6, report.Significance.Helped);
            Assert.Equal(1, report.Significance.Hurt);
        }

        private static SkillEvalReport.CaseResult Case(string id, bool onPass, bool offPass)
        {
            var run = new SkillRunResult("out", null, 1, 1.0);
            return new SkillEvalReport.CaseResult(
                new SkillEvalCase(id, "p", true, []),
                run, [], onPass,
                run, [], offPass,
                TriggerCorrect: true);
        }
    }
}
