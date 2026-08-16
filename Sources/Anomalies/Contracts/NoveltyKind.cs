// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// Whether a peer deviation is something that has just started, or a difference this replica has held
    /// since it came up.
    ///
    /// <para><b>A structured field rather than words in a reason string</b>, for the same argument that makes
    /// <c>SignalFinding.Magnitude</c> a field: a consumer that wants to filter, badge or route on it has to
    /// read it, and parsing prose is not reading.</para>
    ///
    /// <para><b><see cref="New"/> is deliberately the zero value.</b> Everything that has not been through the
    /// novelty gate — every other detector family, every restored incident, every test that predates this —
    /// therefore reports <see cref="New"/>, which is the fail-open answer. A gate whose "unknown" reads as
    /// <see cref="Standing"/> would suppress on missing information, which is the defect this mechanism
    /// replaces rather than a version of the fix.</para>
    /// </summary>
    public enum NoveltyKind
    {
        /// <summary>
        /// Either the deviation is genuinely changing, or there is not yet enough history to say. Reported
        /// every cycle at full severity.
        /// </summary>
        New = 0,

        /// <summary>
        /// The separation from the group has been measured and is not growing — a fixed configuration
        /// difference, a permanently heavier role, a leader. Reported on a slow cadence at reduced severity,
        /// never silenced entirely.
        /// </summary>
        Standing = 1,
    }
}
