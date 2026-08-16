// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// <see cref="ITwoSampleComparer"/> backed by <see cref="MannWhitneyU"/> — the default comparer, and the
    /// right default: rank-based, so it assumes nothing about the distribution, which matters because latency
    /// and per-request cost are heavy-tailed and a mean/σ test on them either fires constantly or misses real
    /// regressions.
    ///
    /// <para>Stateless and thread-safe; use <see cref="Instance"/> rather than allocating per call. The static
    /// class underneath stays public for callers that want the test-specific diagnostics (U, z) or the
    /// pre-sorted and caller-scratch entry points.</para>
    /// </summary>
    public sealed class MannWhitneyComparer : ITwoSampleComparer
    {
        /// <summary>Shared instance — the type holds no state.</summary>
        public static readonly MannWhitneyComparer Instance = new();

        /// <inheritdoc/>
        public string Name => "mann-whitney-u";

        /// <inheritdoc/>
        public TwoSampleComparison Compare(ReadOnlySpan<double> baseline, ReadOnlySpan<double> candidate)
            => MannWhitneyU.Compare(baseline, candidate).ToComparison();

        /// <inheritdoc/>
        public TwoSampleComparison CompareHistograms(
            ReadOnlySpan<long> baselineCounts,
            ReadOnlySpan<long> candidateCounts)
            => MannWhitneyU.CompareHistograms(baselineCounts, candidateCounts).ToComparison();
    }
}
