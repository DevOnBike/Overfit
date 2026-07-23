// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Server.AspNet.Services
{
    /// <summary>
    /// A point-in-time snapshot of the session pool, for the <c>/metrics</c> gauges: how many sessions exist,
    /// how many are decoding right now, how many are free, and how many requests have been shed with 503.
    /// </summary>
    public readonly record struct PoolStatus(int Size, int Active, int Available, long RejectedTotal, int PeakActive);
}
