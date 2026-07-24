// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// Which way a peer departs from the rest of its group.
    ///
    /// <para>Both directions are tested, and not out of completeness: a relative method has no external
    /// reference, so it cannot by itself distinguish "eight members regressed" from "two members are unusually
    /// idle". Testing only the high side would let the first case read as healthy — the group's norm having
    /// quietly moved with it. Reporting the direction keeps the finding true; deciding which side is the fault
    /// needs history, which is the trend/baseline detector's job, not this one's.</para>
    /// </summary>
    public enum PeerDeviation
    {
        /// <summary>Consistent with the rest of the group.</summary>
        None = 0,

        /// <summary>Materially and significantly above its peers — for a cost signal, the suspicious side.</summary>
        High = 1,

        /// <summary>Materially and significantly below its peers.</summary>
        Low = 2,
    }
}
