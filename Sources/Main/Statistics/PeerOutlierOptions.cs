// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Thresholds for a peer-group evaluation. Three knobs with plain meanings — an effect worth acting on, a
    /// confidence level, and a data floor — rather than a "sigma multiplier", which would promise a
    /// normal-distribution trade-off that latency and resource data cannot honour.
    /// </summary>
    /// <param name="MaxPValue">Family-wise significance level <b>before</b> correction; the detector divides it
    /// by the number of peers (Bonferroni), because it runs one test per member.</param>
    /// <param name="MinEffectSize">Smallest Cliff's delta worth reporting. Conventional bands: 0.147 small,
    /// 0.33 medium, 0.474 large.</param>
    /// <param name="MinimumSamplesPerPeer">Usable observations each peer must contribute.</param>
    /// <param name="MinimumPeers">Group members required. Below three there is no "rest of the group" to be an
    /// outlier from — with two members the comparison is symmetric and names no culprit.</param>
    /// <param name="MinRelativeGap">
    /// How far a member's median must sit from its peers', as a fraction of theirs, before the deviation is
    /// worth reporting. Zero disables the gate.
    ///
    /// <para><b>This exists because <see cref="MinEffectSize"/> cannot express "materially different", and
    /// that was measured rather than argued.</b> Cliff's delta is scale-free: it counts how often one
    /// distribution sits above another and says nothing about by how much. Four replicas with medians 860,
    /// 880, 902 and 875 ms — a 3% spread — produced deltas of 0.52 and 0.68 against the 0.33 "medium effect"
    /// gate, so the detector called a perfectly healthy group split and returned <c>Inconclusive</c>. On the
    /// cluster lab that is why a replica genuinely 2.75x slower, with no distribution overlap at all, yielded
    /// no finding.</para>
    ///
    /// <para>The remedy is the shape <see cref="TrendDetector"/> already uses: a rank measure for
    /// <i>consistency</i> and a second gate for <i>size</i>, in the metric's own units. A deviation must clear
    /// both.</para>
    ///
    /// <para>The defaults come from the tightest pair the existing suite pins: an ordinary spread of 4.7% must
    /// be rejected and a real 13.3% regression must be caught, so 8% sits between them with roughly equal
    /// margin on each side. It scales with the profile for the same reason the effect size does.</para>
    /// </param>
    public readonly record struct PeerOutlierOptions(
        double MaxPValue,
        double MinEffectSize,
        int MinimumSamplesPerPeer,
        int MinimumPeers,
        double MinRelativeGap = 0.08)
    {
        /// <summary>
        /// Balanced defaults: 5% family-wise, at least a medium effect, 30 samples per peer (15 minutes at a
        /// 30-second scrape), 3 peers.
        ///
        /// <para><b>Use this, not <c>default</c>.</b> A <c>default(PeerOutlierOptions)</c> is all zeros, which
        /// would demand p ≤ 0 and accept any effect — it is not a usable configuration, and the detector
        /// rejects it.</para>
        /// </summary>
        public static PeerOutlierOptions Balanced => new(0.05, 0.33, 30, 3, 0.08);

        /// <summary>Production profile: same confidence, but only large deviations and a longer window.</summary>
        public static PeerOutlierOptions Strict => new(0.01, 0.474, 60, 3, 0.20);

        /// <summary>Staging profile: reacts on less data, so it demands a bigger effect to compensate.</summary>
        public static PeerOutlierOptions FastFeedback => new(0.05, 0.60, 15, 3, 0.25);

        /// <summary>Whether these thresholds are usable at all — guards the all-zero <c>default</c> trap.</summary>
        public bool IsValid
            => MaxPValue > 0.0
               && MaxPValue <= 1.0
               && MinEffectSize > 0.0
               && MinEffectSize <= 1.0
               && MinimumSamplesPerPeer > 0
               && MinimumPeers >= 3
               && MinRelativeGap >= 0.0;
    }
}
