// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Outcome of a one-sided Mann-Whitney U comparison of a candidate sample against a baseline.
    ///
    /// <para>Both a significance and an effect measure are returned on purpose: with enough samples any
    /// difference becomes "significant", so a decision gate must require a meaningful
    /// <see cref="ProbabilitySuperior"/> / <see cref="CliffsDelta"/> as well as a small
    /// <see cref="PValueCandidateGreater"/>.</para>
    /// </summary>
    /// <param name="U">The U statistic for the candidate sample: the number of (baseline, candidate) pairs in
    /// which the candidate is larger, counting ties as one half. Ranges 0 … n_baseline × n_candidate.</param>
    /// <param name="Z">Normal-approximation z score of <paramref name="U"/> (tie-corrected, with a continuity
    /// correction). 0 when the comparison carries no information.</param>
    /// <param name="PValueCandidateGreater">One-sided p-value for "the candidate is stochastically greater
    /// than the baseline" — i.e. worse, when the metric is a cost such as latency or CPU per request.</param>
    /// <param name="ProbabilitySuperior">Common-language effect size: the probability that a randomly drawn
    /// candidate value exceeds a randomly drawn baseline value (ties count as one half). 0.5 = no difference,
    /// 1.0 = the candidate is always worse.</param>
    /// <param name="CliffsDelta">Effect size on −1 … +1 (<c>2 × ProbabilitySuperior − 1</c>). 0 = no
    /// difference; positive = the candidate is stochastically greater.</param>
    /// <param name="BaselineCount">Number of baseline observations.</param>
    /// <param name="CandidateCount">Number of candidate observations.</param>
    public readonly record struct MannWhitneyResult(
        double U,
        double Z,
        double PValueCandidateGreater,
        double ProbabilitySuperior,
        double CliffsDelta,
        int BaselineCount,
        int CandidateCount)
    {
        /// <summary>
        /// Projects onto the test-agnostic <see cref="TwoSampleComparison"/> a decision gate consumes, dropping
        /// <see cref="U"/> and <see cref="Z"/> — those are diagnostics of this particular test and a rollout
        /// decision has no business depending on them.
        /// </summary>
        public TwoSampleComparison ToComparison()
        {
            return new TwoSampleComparison(PValueCandidateGreater, CliffsDelta, ProbabilitySuperior, BaselineCount, CandidateCount);
        }
    }
}
