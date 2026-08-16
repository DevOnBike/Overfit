// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>
    /// The single status vocabulary every detector answers in. Shared on purpose: an incident model that lets
    /// each detector invent its own states cannot group, deduplicate or route them, and the distinctions below
    /// are exactly the ones that stop a monitoring system from quietly lying.
    ///
    /// <para>The load-bearing separation is between <see cref="Healthy"/> and the two undecidable states.
    /// "Nothing is wrong" and "we could not tell" must never collapse into the same answer — a detector with no
    /// data reporting health is worse than no detector.</para>
    /// </summary>
    public enum DetectionStatus
    {
        /// <summary>Evaluated, and nothing deviates.</summary>
        Healthy = 0,

        /// <summary>Evaluated, and something deviates significantly and materially.</summary>
        Anomalous = 1,

        /// <summary>
        /// Evaluated, but the evidence points in contradictory directions — a peer group with no coherent norm,
        /// a series that both rises and falls within the window. Reported rather than resolved by fiat.
        /// </summary>
        Inconclusive = 2,

        /// <summary>
        /// Not enough to decide, and not merely early: too few members, a required companion series absent,
        /// or nothing usable after filtering. Never to be treated as healthy.
        /// </summary>
        InsufficientData = 3,

        /// <summary>
        /// Data is arriving but the window is not yet long enough. Distinct from
        /// <see cref="InsufficientData"/> because it resolves itself with time and should not be reported as a
        /// configuration problem.
        /// </summary>
        WarmingUp = 4,
    }
}
