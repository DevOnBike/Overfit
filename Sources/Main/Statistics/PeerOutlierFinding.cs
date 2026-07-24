// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// One member's standing against the rest of its group. Produced for <b>every</b> peer, not only the
    /// deviating ones — an explanation that shows the whole group is checkable, whereas a bare "pod-7 is bad"
    /// is something the reader has to take on trust.
    /// </summary>
    /// <param name="Name">The peer, copied from <see cref="PeerSeries.Name"/>.</param>
    /// <param name="Comparison">
    /// The decisive test: this peer's window against the pooled windows of all the others. For
    /// <see cref="PeerDeviation.Low"/> the arms are swapped, so <c>EffectSize</c> reads as the magnitude of the
    /// departure in that direction rather than as a negative number.
    /// </param>
    /// <param name="Deviation">Which way it departs, if at all.</param>
    /// <param name="UsableSamples">Observations that survived filtering — non-finite values, and samples with
    /// non-positive work on a load-sensitive signal, are dropped.</param>
    public readonly record struct PeerOutlierFinding(
        string Name,
        TwoSampleComparison Comparison,
        PeerDeviation Deviation,
        int UsableSamples)
    {
        /// <summary>Whether this member departs from the group in either direction.</summary>
        public bool IsOutlier => Deviation != PeerDeviation.None;
    }
}
