// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// How confidently a channel could be bound.
    ///
    /// <para>The three values are deliberately distinct, because they call for three different responses and
    /// collapsing any two of them loses the whole point of running discovery: one is done, one needs a
    /// decision, and one needs an exporter.</para>
    /// </summary>
    public enum DiscoveryOutcome : byte
    {
        /// <summary>
        /// No series matched, or the ones that matched exist in the cluster and are not exported by these
        /// pods. The guard will be blind on this channel — knowingly, which is the improvement.
        /// </summary>
        NotFound = 0,

        /// <summary>Exactly one evidenced candidate. Bind it.</summary>
        Resolved = 1,

        /// <summary>
        /// Several evidenced candidates, and nothing in the data says which is right.
        ///
        /// <para><b>A human has to choose, and this is not a gap in the implementation.</b> The clearest case
        /// is the error rate: whether a 4xx counts as an error is a business decision, not a property of the
        /// series, and two exporters can both be present and both be plausible. Guessing here would produce a
        /// guard that is confidently measuring the wrong thing.</para>
        /// </summary>
        Ambiguous = 2,
    }
}
