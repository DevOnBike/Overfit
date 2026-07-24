// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// The test-agnostic verdict a decision gate consumes: how confident the difference is, how big it is, and
    /// how much data it rests on. Individual tests keep their own diagnostics (Mann-Whitney's U and z live on
    /// <see cref="MannWhitneyResult"/>); this carries only what a rollout decision may legitimately depend on.
    /// </summary>
    /// <param name="PValueCandidateWorse">One-sided p-value for "the candidate is stochastically worse".</param>
    /// <param name="EffectSize">Signed magnitude on −1 … +1; positive means the candidate is worse. Distribution-
    /// free, so it stays comparable across metrics with wildly different units.</param>
    /// <param name="ProbabilityCandidateWorse">Probability that a random candidate observation is worse than a
    /// random baseline one (ties as one half). 0.5 = no difference.</param>
    /// <param name="BaselineCount">Observations behind the baseline arm.</param>
    /// <param name="CandidateCount">Observations behind the candidate arm.</param>
    public readonly record struct TwoSampleComparison(
        double PValueCandidateWorse,
        double EffectSize,
        double ProbabilityCandidateWorse,
        int BaselineCount,
        int CandidateCount)
    {
        /// <summary>
        /// The gate itself: a regression must be <b>significant AND materially large AND backed by enough
        /// data</b>. All three are required on purpose.
        ///
        /// <para>Dropping the effect size means large windows roll back on noise — with enough samples any
        /// difference reaches p &lt; 0.05. Dropping the sample floor means a handful of requests can produce a
        /// confident-looking verdict that the next minute contradicts; when this returns false for want of
        /// data, the answer to report is "not enough evidence", never "healthy".</para>
        /// </summary>
        /// <param name="maxPValue">Significance level, e.g. 0.05 — already corrected if several metrics are tested.</param>
        /// <param name="minEffectSize">Smallest effect worth acting on, e.g. 0.15 (below Cliff's "small" boundary
        /// of 0.147 is conventionally negligible).</param>
        /// <param name="minimumSamplesPerArm">Floor on both arms; below it there is no verdict to give.</param>
        public bool IsRegression(double maxPValue, double minEffectSize, int minimumSamplesPerArm)
        {
            if (!HasEnoughData(minimumSamplesPerArm))
            {
                return false;
            }

            return PValueCandidateWorse <= maxPValue && EffectSize >= minEffectSize;
        }

        /// <summary>
        /// Whether both arms cleared the sample floor. Distinguishes "no regression" from "no evidence" — the
        /// difference between a green deploy and a canary that never received traffic.
        /// </summary>
        public bool HasEnoughData(int minimumSamplesPerArm)
        {
            return BaselineCount >= minimumSamplesPerArm && CandidateCount >= minimumSamplesPerArm;
        }
    }
}
