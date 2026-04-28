// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly struct DotNetMemorySnapshot
    {
        public readonly ValueStopwatch Now;

        public readonly long TotalAllocatedBytes;
        public readonly long LiveManagedBytes;
        public readonly long WorkingSetBytes;
        public readonly long PrivateMemoryBytes;

        public readonly int Gen0Collections;
        public readonly int Gen1Collections;
        public readonly int Gen2Collections;

        public DotNetMemorySnapshot(Process process)
        {
            Now = ValueStopwatch.StartNew();

            TotalAllocatedBytes = GC.GetTotalAllocatedBytes();
            LiveManagedBytes = GC.GetTotalMemory(false);

            WorkingSetBytes = process.WorkingSet64;
            PrivateMemoryBytes = process.PrivateMemorySize64;

            Gen0Collections = GC.CollectionCount(0);
            Gen1Collections = GC.CollectionCount(1);
            Gen2Collections = GC.CollectionCount(2);
        }
    }
}
