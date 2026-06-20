// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.Diagnostics.Tracing
{
    internal sealed class DiagnosticsTraceEntryDiff
    {
        public string Name { get; set; } = string.Empty;
        public long CountDelta
        {
            get; set;
        }
        public double DurationMsDelta
        {
            get; set;
        }
        public long AllocatedBytesDelta
        {
            get; set;
        }
        public double CurrentDurationMs
        {
            get; set;
        }
        public long CurrentAllocatedBytes
        {
            get; set;
        }
    }
}
