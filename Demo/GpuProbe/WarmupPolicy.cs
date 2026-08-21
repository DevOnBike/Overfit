// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The rule that decides when an arm has finished warming up. It is a STOPPING RULE, not a count.
    /// <para>
    /// Measured on 2026-08-21 on a 9950X3D, --quick shapes. At the shipped default of 2 warm-ups the C3
    /// host arm reads 0.560 ms against a settled median of 0.145 ms - a factor of about four, from
    /// warm-up alone. The cold start is larger than that figure suggests at the very first readings:
    /// with the warm-up capped at ten rounds, C3's first five readings had a median of 10.270 ms against
    /// 0.891 ms for the next five, and the canary's were 97.186 ms against 5.328 ms.
    /// </para>
    /// <para>
    /// A CORRECTION worth keeping, because it is the more useful lesson. This comment first justified the
    /// rule with 6.841 ms, 0.007 ms and a ratio "inflated by about a million". Those readings are real but
    /// they came from a DELIBERATELY MUTATED build with the device barrier removed, whose report artefact
    /// was left in the working tree and committed. Warm-up is worth about 4x here; the missing barrier was
    /// a different defect worth about 8000x. Twenty-five warm-ups happened to be enough on that box, and a
    /// fixed 25 would still be a number that worked once on one box - so the probe warms until the
    /// measurement stops moving and prints how many rounds that took.
    /// </para>
    /// <para>
    /// The rule: keep a sliding window of the last <see cref="WindowSize"/> warm-up readings of an arm.
    /// After each round compare the median of that window against the median of the window before it. An
    /// arm is SETTLED when the two medians differ by no more than <see cref="Tolerance"/> of the earlier
    /// one. Medians, because a single reading is not a fact; two adjacent windows, because a first
    /// derivative is what "has stopped moving" means and an absolute threshold would need a unit.
    /// </para>
    /// </summary>
    internal sealed class WarmupPolicy
    {
        public WarmupPolicy(int minRounds, int maxRounds, double tolerance, int windowSize, double budgetMs)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(windowSize, 3);
            ArgumentOutOfRangeException.ThrowIfLessThan(minRounds, 2 * windowSize);
            ArgumentOutOfRangeException.ThrowIfLessThan(maxRounds, minRounds);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(tolerance, 0);

            MinRounds = minRounds;
            MaxRounds = maxRounds;
            Tolerance = tolerance;
            WindowSize = windowSize;
            BudgetMs = budgetMs;
        }

        /// <summary>Readings per window. Two windows must exist before the rule can be evaluated.</summary>
        public int WindowSize { get; }

        /// <summary>
        /// Rounds that always run, whatever the readings say. It is at least twice the window so the rule
        /// is evaluable at the first opportunity, and it is the floor that replaces the old fixed count.
        /// </summary>
        public int MinRounds { get; }

        /// <summary>Hard cap. The loop always terminates, settled or not, and says which happened.</summary>
        public int MaxRounds { get; }

        /// <summary>Relative move between the two window medians that still counts as settled.</summary>
        public double Tolerance { get; }

        /// <summary>
        /// Wall-clock budget for the warm-up phase of one cell. Checked only AFTER
        /// <see cref="MinRounds"/>, so a cell too slow to fit its budget still produces a verdict
        /// instead of no verdict at all.
        /// </summary>
        public double BudgetMs { get; }

        public static WarmupPolicy Default => new(minRounds: 10, maxRounds: 100, tolerance: 0.05, windowSize: 5, budgetMs: 30_000);

        public string Describe() => string.Create(
            CultureInfo.InvariantCulture,
            $"warm up until the median of the last {WindowSize} readings of an arm differs from the median of " +
            $"the {WindowSize} before it by no more than {Tolerance * 100:F1} %, or by no more than the arm's own " +
            $"two-sigma scatter band, whichever is larger; at least {MinRounds} rounds, " +
            $"at most {MaxRounds}, and at most {BudgetMs / 1000:F0} s per cell after the minimum is reached");
    }
}
