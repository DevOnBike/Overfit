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
    public readonly record struct PeerOutlierOptions(
        double MaxPValue,
        double MinEffectSize,
        int MinimumSamplesPerPeer,
        int MinimumPeers)
    {
        /// <summary>
        /// Balanced defaults: 5% family-wise, at least a medium effect, 30 samples per peer (15 minutes at a
        /// 30-second scrape), 3 peers.
        ///
        /// <para><b>Use this, not <c>default</c>.</b> A <c>default(PeerOutlierOptions)</c> is all zeros, which
        /// would demand p ≤ 0 and accept any effect — it is not a usable configuration, and the detector
        /// rejects it.</para>
        /// </summary>
        public static PeerOutlierOptions Balanced => new(0.05, 0.33, 30, 3);

        /// <summary>Production profile: same confidence, but only large deviations and a longer window.</summary>
        public static PeerOutlierOptions Strict => new(0.01, 0.474, 60, 3);

        /// <summary>Staging profile: reacts on less data, so it demands a bigger effect to compensate.</summary>
        public static PeerOutlierOptions FastFeedback => new(0.05, 0.60, 15, 3);

        /// <summary>Whether these thresholds are usable at all — guards the all-zero <c>default</c> trap.</summary>
        public bool IsValid
            => MaxPValue > 0.0
               && MaxPValue <= 1.0
               && MinEffectSize > 0.0
               && MinEffectSize <= 1.0
               && MinimumSamplesPerPeer > 0
               && MinimumPeers >= 3;
    }
}
