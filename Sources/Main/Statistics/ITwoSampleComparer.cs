// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// "Are these two populations different, beyond noise?" — the one operation a comparison-based analyser
    /// rests on, behind an abstraction so the decision policy does not depend on which test answers it.
    ///
    /// <para>This exists because the question outlives the answer. The same comparison drives a canary
    /// (baseline version versus new version), a peer-group outlier check (one replica versus its siblings) and
    /// a before/after change check — only the axis of the split differs. Alternative statistics get proposed
    /// regularly, so the decision layer is written against this interface and tested with a stub, rather than
    /// welded to a particular test.</para>
    ///
    /// <para>Both ingestion shapes are part of the contract, not optional extras: raw per-request values are
    /// often unavailable in a customer's cluster, where the metric backend has already reduced everything to
    /// histogram buckets. A comparer that cannot consume buckets is not deployable.</para>
    /// </summary>
    public interface ITwoSampleComparer
    {
        /// <summary>Stable identifier for reports and exported metric labels, e.g. <c>mann-whitney-u</c>.</summary>
        string Name
        {
            get;
        }

        /// <summary>
        /// Compares raw observations. Higher values must mean <i>worse</i> (latency, cost, error count); for a
        /// metric where higher is better, negate both samples before calling.
        /// </summary>
        TwoSampleComparison Compare(ReadOnlySpan<double> baseline, ReadOnlySpan<double> candidate);

        /// <summary>
        /// Compares bucketed counts over shared, ascending bucket boundaries — the shape a scraped histogram
        /// arrives in. Resolution is bounded by the bucket widths: a difference smaller than a boundary is
        /// genuinely absent from the input and will be reported as no difference.
        /// </summary>
        TwoSampleComparison CompareHistograms(
            ReadOnlySpan<long> baselineCounts,
            ReadOnlySpan<long> candidateCounts);
    }
}
