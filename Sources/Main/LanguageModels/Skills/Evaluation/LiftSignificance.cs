// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Turns a raw <see cref="SkillEvalReport.Lift"/> into a claim you can defend: a confidence interval and a
    /// p-value. "+20%" on 10 cases is one lucky case; this says whether the lift survives the sample size.
    ///
    /// <para>The design is <b>paired</b> — the same case is run skill-ON and skill-OFF against the same model and
    /// seed — so the correct test is <b>McNemar's</b> on the discordant pairs, not a two-independent-proportions
    /// test (which would throw away the pairing and overstate the variance). Only cases where the two arms
    /// DISAGREE carry information: <paramref name="Helped"/> (ON passed, OFF failed) and <paramref name="Hurt"/>
    /// (ON failed, OFF passed). Cases where both arms agree cancel out — and indeed
    /// <c>PassRateOn − PassRateOff == (Helped − Hurt) / total</c> exactly, so this is the same lift, now with error bars.</para>
    ///
    /// <para>The p-value is an <b>exact</b> two-sided binomial test (X ~ Binomial(Helped+Hurt, 0.5)) — the right
    /// choice for the 10–20-case eval sets this harness is built for, where the chi-square approximation is poor.
    /// The interval is the Wald interval for the difference of correlated proportions; at small N treat it as
    /// indicative, not gospel.</para>
    /// </summary>
    /// <param name="Lift">Paired difference in pass rate, <c>(Helped − Hurt) / total</c>, in [-1, 1].</param>
    /// <param name="LowerBound">Lower end of the 95% interval on <paramref name="Lift"/>.</param>
    /// <param name="UpperBound">Upper end of the 95% interval on <paramref name="Lift"/>.</param>
    /// <param name="PValue">Two-sided probability of a lift this extreme if the skill did nothing.</param>
    /// <param name="IsSignificant"><c>PValue &lt; 0.05</c>. STATISTICAL only — it says the lift is probably real,
    /// NOT that it is big enough to matter. Apply your own magnitude threshold on top.</param>
    /// <param name="Helped">Discordant pairs where the skill turned a fail into a pass.</param>
    /// <param name="Hurt">Discordant pairs where the skill turned a pass into a fail.</param>
    public sealed record LiftSignificance(
        double Lift,
        double LowerBound,
        double UpperBound,
        double PValue,
        bool IsSignificant,
        int Helped,
        int Hurt)
    {
        /// <summary>z for a two-sided 95% interval.</summary>
        private const double Z95 = 1.959963985;

        /// <summary>Above this many discordant pairs the exact tail underflows; fall back to the normal
        /// approximation. Unreachable for realistic eval sets (10–20 cases) — it just keeps the function total.</summary>
        private const int ExactLimit = 1000;

        /// <summary>
        /// Computes the interval + p-value from the discordant-pair counts. Pure and total: a zero-case or
        /// fully-concordant eval yields a zero lift, a zero-width interval and <c>p = 1</c> (no evidence), never
        /// a divide-by-zero or a false "significant".
        /// </summary>
        /// <param name="helped">Cases where ON passed and OFF failed.</param>
        /// <param name="hurt">Cases where ON failed and OFF passed.</param>
        /// <param name="total">Total cases evaluated (concordant ones included — they set the scale).</param>
        public static LiftSignificance Compute(int helped, int hurt, int total)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(helped);
            ArgumentOutOfRangeException.ThrowIfNegative(hurt);
            ArgumentOutOfRangeException.ThrowIfNegative(total);

            var discordant = helped + hurt;
            if (total == 0 || discordant == 0)
            {
                // No cases, or every case agreed: the arms are indistinguishable. Lift is exactly 0 and there is
                // no evidence of an effect — p = 1, not "significant".
                return new LiftSignificance(0.0, 0.0, 0.0, 1.0, false, helped, hurt);
            }

            var n = (double)total;
            var delta = helped - hurt;
            var lift = delta / n;

            // Wald variance for the difference of PAIRED proportions: (b + c − (b − c)²/n) / n².
            var variance = (discordant - ((double)delta * delta / n)) / (n * n);
            var half = Z95 * Math.Sqrt(Math.Max(variance, 0.0));
            var p = McNemarPValue(helped, hurt);

            return new LiftSignificance(
                lift,
                Math.Clamp(lift - half, -1.0, 1.0),
                Math.Clamp(lift + half, -1.0, 1.0),
                p,
                p < 0.05,
                helped,
                hurt);
        }

        /// <summary>Two-sided exact binomial (McNemar) p: under "the skill does nothing", each discordant pair is
        /// a fair coin, so p = 2·P(X ≥ max(b,c)) for X ~ Binomial(b+c, 0.5), capped at 1.</summary>
        private static double McNemarPValue(int helped, int hurt)
        {
            var n = helped + hurt;
            var k = Math.Max(helped, hurt);

            if (n > ExactLimit)
            {
                // Continuity-corrected normal approximation; only for absurdly large evals.
                var z = (Math.Abs(helped - hurt) - 1.0) / Math.Sqrt(n);

                return Math.Clamp(Erfc(Math.Max(z, 0.0) / Math.Sqrt(2.0)), 0.0, 1.0);
            }

            // Walk the PMF by ratio from PMF(0) = 0.5^n — no factorials, no overflow.
            var pmf = Math.Pow(0.5, n);
            var tail = 0.0;

            for (var i = 0; i <= n; i++)
            {
                if (i >= k)
                {
                    tail += pmf;
                }

                pmf = pmf * (n - i) / (i + 1);
            }

            return Math.Min(1.0, 2.0 * tail);
        }

        /// <summary>Complementary error function — Numerical Recipes' rational approximation (|error| ≲ 1.2e-7),
        /// which is far tighter than any p-value threshold needs.</summary>
        private static double Erfc(double x)
        {
            var z = Math.Abs(x);
            var t = 1.0 / (1.0 + (0.5 * z));

            // Horner, unrolled iteratively — the deeply-nested one-liner form of this is unreadable and easy to
            // mis-parenthesise.
            var poly = -0.82215223 + (t * 0.17087277);
            poly = 1.48851587 + (t * poly);
            poly = -1.13520398 + (t * poly);
            poly = 0.27886807 + (t * poly);
            poly = -0.18628806 + (t * poly);
            poly = 0.09678418 + (t * poly);
            poly = 0.37409196 + (t * poly);
            poly = 1.00002368 + (t * poly);

            var ans = t * Math.Exp((-z * z) - 1.26551223 + (t * poly));

            return x >= 0.0 ? ans : 2.0 - ans;
        }
    }
}
