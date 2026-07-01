// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Result of a skill eval run: the per-case verdicts plus the aggregates that actually matter — the
    /// skill-ON pass rate, the skill-OFF (unaided) pass rate, their difference (<see cref="Lift"/>, the real
    /// signal — a bare score is noise), and trigger accuracy. Persist and compare across runs; the delta
    /// between runs is the regression/improvement signal.
    /// </summary>
    public sealed class SkillEvalReport
    {
        /// <summary>Per-case outcome: the same prompt graded with the skill ON and OFF.</summary>
        public sealed record CaseResult(
            SkillEvalCase Case,
            SkillRunResult OnResult,
            IReadOnlyList<GradeCheck> OnChecks,
            bool OnPass,
            SkillRunResult OffResult,
            IReadOnlyList<GradeCheck> OffChecks,
            bool OffPass,
            bool TriggerCorrect);

        public IReadOnlyList<CaseResult> Cases { get; }

        /// <summary>Fraction of cases that pass all their checks with the skill enabled.</summary>
        public double PassRateOn { get; }

        /// <summary>Fraction that pass with the skill disabled (the unaided-model baseline).</summary>
        public double PassRateOff { get; }

        /// <summary>The skill's real value: <see cref="PassRateOn"/> − <see cref="PassRateOff"/>. Near zero
        /// means the bare model already does this — a candidate to retire.</summary>
        public double Lift => PassRateOn - PassRateOff;

        /// <summary>Fraction of cases where the skill (de)activated as expected (<c>ShouldTrigger</c>).</summary>
        public double TriggerAccuracy { get; }

        public SkillEvalReport(IReadOnlyList<CaseResult> cases)
        {
            ArgumentNullException.ThrowIfNull(cases);
            Cases = cases;

            if (cases.Count == 0)
            {
                return;
            }

            var on = 0;
            var off = 0;
            var trigger = 0;
            for (var i = 0; i < cases.Count; i++)
            {
                if (cases[i].OnPass)
                {
                    on++;
                }
                if (cases[i].OffPass)
                {
                    off++;
                }
                if (cases[i].TriggerCorrect)
                {
                    trigger++;
                }
            }

            PassRateOn = on / (double)cases.Count;
            PassRateOff = off / (double)cases.Count;
            TriggerAccuracy = trigger / (double)cases.Count;
        }
    }
}
