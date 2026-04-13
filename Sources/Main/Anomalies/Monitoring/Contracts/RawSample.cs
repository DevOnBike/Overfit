// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public readonly record struct RawSample
    {
        public long Timestamp { get; init; }
        public float Value { get; init; }
    }
}