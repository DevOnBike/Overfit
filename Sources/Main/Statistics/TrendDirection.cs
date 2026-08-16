// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Statistics
{
    /// <summary>Which way a series is moving over the observed window.</summary>
    public enum TrendDirection
    {
        /// <summary>No monotone movement worth reporting.</summary>
        None = 0,

        /// <summary>Climbing — for a cost signal, the direction that ends at a limit.</summary>
        Rising = 1,

        /// <summary>Falling. Reported rather than ignored: a collapsing request rate or cache hit ratio is a
        /// symptom, and a detector that only watches the upside misses half of them.</summary>
        Falling = 2,
    }
}
