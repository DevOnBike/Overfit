// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Serving
{
    /// <summary>
    /// A point-in-time snapshot of an <see cref="OverfitResourcePool{T}"/>'s load — the numbers a server exports
    /// (e.g. as Prometheus gauges/counters) so an operator can see saturation and load-shedding.
    /// </summary>
    /// <param name="Size">Pool size (maximum concurrency).</param>
    /// <param name="Active">Items currently checked out.</param>
    /// <param name="Available">Items free to rent right now.</param>
    /// <param name="TotalRented">Lifetime successful rentals.</param>
    /// <param name="TotalRejected">Lifetime rentals refused because no item freed up within the timeout (shed load).</param>
    /// <param name="PeakActive">Highest concurrent rentals observed.</param>
    /// <param name="MeanWaitMs">Mean time a rental waited for a free slot, in milliseconds.</param>
    public readonly record struct PoolMetrics(
        int Size,
        int Active,
        int Available,
        long TotalRented,
        long TotalRejected,
        int PeakActive,
        double MeanWaitMs);
}
