// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using DevOnBike.Overfit.Runtime;

// Hammers OverfitParallel.For and keeps a ledger of which indices ran, so a chunk executed twice or not at
// all is reported with the dispatch that produced it.
//
// **Why this exists.** `XC-97` is a dispatcher defect that only shows at chunksPerWorker > 1, and the only
// instrument for it was the full test suite: thirty seconds when healthy, and when the defect fires it takes
// the host down without printing a failure, so a run ends at 176 of 2748 tests reporting "Passed!". Debugging
// a claim protocol through that is how an afternoon disappears — 2026-08-19 is on record for it.
//
// This runs thousands of dispatches per second, checks every one, and stops at the first bad dispatch with
// its shape printed. It is deliberately a separate executable rather than a test: a defect that kills the
// process must not be able to take a suite run with it.
//
//   dotnet run -c Release --project Scripts/DispatchStress -- [iterations]
internal static class DispatchStress
{
    private static unsafe void Body(int start, int end, void* context)
    {
        ref readonly var ledger = ref Unsafe.AsRef<Ledger>(context);

        for (var i = start; i < end; i++)
        {
            if ((uint)i >= (uint)ledger.Length)
            {
                Interlocked.Increment(ref *ledger.OutOfRange);

                continue;
            }

            Interlocked.Increment(ref ledger.Counts[i]);
        }
    }

    private readonly unsafe struct Ledger
    {
        public readonly int* Counts;
        public readonly int* OutOfRange;
        public readonly int Length;

        public Ledger(int* counts, int* outOfRange, int length)
        {
            Counts = counts;
            OutOfRange = outOfRange;
            Length = length;
        }
    }

    private static unsafe int Main(string[] args)
    {
        var iterations = args.Length > 0 ? int.Parse(args[0]) : 2000;

        Console.WriteLine($"workers {OverfitParallel.WorkerCount}, "
                          + $"chunk factor {OverfitParallel.ChunkFactor}, "
                          + $"iterations {iterations}");

        // Lengths that straddle the worker count and the chunk table, plus the awkward small ones.
        // A single shape can be pinned from the command line. That is what separates "this shape is
        // broken" from "this shape is broken only after other dispatches have run", and those are different
        // defects: the first is in the split, the second is state carried between dispatches.
        int[] lengths = args.Length > 1 ? [int.Parse(args[1])]
            : [1, 2, 15, 16, 17, 31, 33, 64, 100, 255, 256, 257, 1000, 4096, 100_000];
        int[] factors = args.Length > 2 ? [int.Parse(args[2])] : [1, 2, 3, 4, 8];

        var clock = Stopwatch.StartNew();
        var dispatches = 0L;

        foreach (var chunksPerWorker in factors)
        {
            foreach (var length in lengths)
            {
                var counts = new int[length];

                for (var iteration = 0; iteration < iterations; iteration++)
                {
                    Array.Clear(counts);
                    var outOfRange = 0;

                    fixed (int* pointer = counts)
                    {
                        var ledger = new Ledger(pointer, &outOfRange, length);

                        OverfitParallel.For(
                            0, length, 1, OverfitParallel.WorkerCount, &Body, &ledger, chunksPerWorker);
                    }

                    dispatches++;

                    var missing = 0;
                    var duplicated = 0;
                    var worst = 0;

                    for (var i = 0; i < length; i++)
                    {
                        if (counts[i] == 1)
                        {
                            continue;
                        }

                        if (counts[i] == 0)
                        {
                            missing++;

                            continue;
                        }

                        duplicated++;

                        if (counts[i] > worst)
                        {
                            worst = counts[i];
                        }
                    }

                    if (missing == 0 && duplicated == 0 && outOfRange == 0)
                    {
                        continue;
                    }

                    Console.WriteLine();
                    Console.WriteLine($"FAILED at dispatch {dispatches}: length {length}, "
                                      + $"{chunksPerWorker} chunks per worker, iteration {iteration}");
                    Console.WriteLine($"  never ran   : {missing}");
                    Console.WriteLine($"  ran twice+  : {duplicated} (worst {worst} times)");
                    Console.WriteLine($"  out of range: {outOfRange}");

                    return 1;
                }
            }
        }

        clock.Stop();

        Console.WriteLine($"clean: {dispatches} dispatches in {clock.Elapsed.TotalSeconds:F1} s "
                          + $"({dispatches / clock.Elapsed.TotalSeconds:F0}/s)");

        return 0;
    }
}
