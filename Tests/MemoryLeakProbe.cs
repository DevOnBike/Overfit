// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;

namespace DevOnBike.Overfit.Tests
{
    internal static class MemoryLeakProbe
    {
        public readonly record struct Snapshot(
            long LiveManagedBytes,
            long TotalAllocatedBytes,
            long WorkingSet64,
            long PrivateMemoryBytes,
            int Gen0Collections,
            int Gen1Collections,
            int Gen2Collections)
        {
            public double LiveManagedMb => LiveManagedBytes / 1024.0 / 1024.0;
            public double TotalAllocatedMb => TotalAllocatedBytes / 1024.0 / 1024.0;
            public double WorkingSetMb => WorkingSet64 / 1024.0 / 1024.0;
            public double PrivateMemoryMb => PrivateMemoryBytes / 1024.0 / 1024.0;
        }

        public static Snapshot Capture(bool forceFullGc)
        {
            if (forceFullGc)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }

            using var process = Process.GetCurrentProcess();

            return new Snapshot(
                LiveManagedBytes: GC.GetTotalMemory(false),
                TotalAllocatedBytes: GC.GetTotalAllocatedBytes(precise: true),
                WorkingSet64: process.WorkingSet64,
                PrivateMemoryBytes: process.PrivateMemorySize64,
                Gen0Collections: GC.CollectionCount(0),
                Gen1Collections: GC.CollectionCount(1),
                Gen2Collections: GC.CollectionCount(2));
        }

        public static string Format(string label, Snapshot snapshot)
        {
            return
                $"=== MEMORY SNAPSHOT: {label} ==={Environment.NewLine}" +
                $"live managed:   {snapshot.LiveManagedMb:F2} MB{Environment.NewLine}" +
                $"total alloc:    {snapshot.TotalAllocatedMb:F2} MB{Environment.NewLine}" +
                $"working set:    {snapshot.WorkingSetMb:F2} MB{Environment.NewLine}" +
                $"private bytes:  {snapshot.PrivateMemoryMb:F2} MB{Environment.NewLine}" +
                $"GC0/1/2 total:  {snapshot.Gen0Collections}/{snapshot.Gen1Collections}/{snapshot.Gen2Collections}";
        }
    }
}