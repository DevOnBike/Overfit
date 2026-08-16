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
    /// <param name="RelativeGap">
    /// How far this member's median sits from its peers', as a fraction of theirs — the magnitude the rank
    /// statistics cannot express.
    ///
    /// <para><b>Reported because neither of the other numbers answers "by how much".</b> Cliff's delta counts
    /// overlap and saturates: two tight distributions three percent apart score the same as two three hundred
    /// percent apart. A p-value is a statement about evidence, not size. An operator handed only those two has
    /// no way to tell a replica worth paging for from one that is a rounding error, and neither does a
    /// downstream gate.</para>
    ///
    /// <para><see cref="double.PositiveInfinity"/> when the peers' median carries no usable scale — a group
    /// centred on zero has no relative distance to report.</para>
    /// </param>
    /// <param name="AbsoluteGap">
    /// The same distance in the signal's own units. Reported alongside the relative one because a percentage
    /// alone hides the case that matters: 14% of a GC-pause ratio of 0.004 is three tenths of a millisecond,
    /// and no percentage makes that worth waking anyone.
    /// </param>
    public readonly record struct PeerOutlierFinding(
        string Name,
        TwoSampleComparison Comparison,
        PeerDeviation Deviation,
        int UsableSamples,
        double RelativeGap,
        double AbsoluteGap)
    {
        /// <summary>Whether this member departs from the group in either direction.</summary>
        public bool IsOutlier => Deviation != PeerDeviation.None;
    }
}
