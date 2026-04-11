// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public enum MetricIndex : byte
    {
        CpuUsageRatio = 0,
        CpuThrottleRatio = 1,
        MemoryWorkingSetBytes = 2,
        OomEventsRate = 3,
        LatencyP50Ms = 4,
        LatencyP95Ms = 5,
        LatencyP99Ms = 6,
        RequestsPerSecond = 7,
        ErrorRate = 8,
        GcGen2HeapBytes = 9,
        GcPauseRatio = 10,
        ThreadPoolQueueLength = 11,

        Count = 12   // sentinel — liczba featurów
    }
}