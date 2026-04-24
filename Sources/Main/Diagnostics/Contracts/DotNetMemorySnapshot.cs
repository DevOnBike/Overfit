// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Diagnostics.Contracts
{
    public readonly struct DotNetMemorySnapshot
    {
        public readonly ValueStopwatch Now;

        public readonly long TotalAllocatedBytes;
        public readonly int Gen0Collections;
        public readonly int Gen1Collections;
        public readonly int Gen2Collections;

        public DotNetMemorySnapshot()
        {
            Now = ValueStopwatch.StartNew();

            TotalAllocatedBytes = GC.GetTotalAllocatedBytes(false);
            Gen0Collections = GC.CollectionCount(0);
            Gen1Collections = GC.CollectionCount(1);
            Gen2Collections = GC.CollectionCount(2);
        }
    }
}
