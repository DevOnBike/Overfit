// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.Diagnostics.Tracing
{
    internal sealed class DiagnosticsTraceDiff
    {
        public int BaselineEpoch
        {
            get; set;
        }
        public int CurrentEpoch
        {
            get; set;
        }
        public double GraphBackwardMsDelta
        {
            get; set;
        }
        public long GraphAllocatedBytesDelta
        {
            get; set;
        }
        public long TapeOpsDelta
        {
            get; set;
        }
        public List<DiagnosticsTraceEntryDiff> ModuleDiffs { get; } = [];
        public List<DiagnosticsTraceEntryDiff> KernelDiffs { get; } = [];
    }
}
