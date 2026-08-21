// Prints OverfitParallel's occupancy report for one ONNX model, because nothing else does.
//
// WHY THIS EXISTS. `OverfitParallel.MeasureOccupancy` is switched on by OVERFIT_PARALLEL_OCCUPANCY=1 and
// records pool use, straggler ratio and dispatch overhead — the three numbers XC-93 used to argue that at
// 100% pool use the 60.9 MB CNN would run 12.27 ms instead of 18.88. But the totals only surface if
// something calls the public OccupancyReport(), and BenchmarkDotNet never will: a BDN run of
// LargeCnnComparisonBenchmark with the variable set prints nothing at all. Measured 2026-08-21, twice.
// So the diagnostic those rows rest on could not be re-taken by anybody, on any build, without this.
//
// WHY IT IS NOT A BENCHMARK. It reports wall time per iteration so the occupancy figures have something to
// sit against, and that is all. BenchmarkDotNet owns timing claims here; this owns the occupancy triple.
// Do not quote its milliseconds as a performance result — run LargeCnnComparisonBenchmark for that.
//
// THE ONE GUARD THAT MATTERS. If occupancy was not actually measured, this exits non-zero rather than
// printing a report that says so in prose. A harness that prints "(occupancy not measured)" and returns 0
// is the same shape as a test run that executes nothing and reports success, and this repository has been
// bitten by that shape more than once.

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Runtime;

internal static class Program
{
    private const int InputSize = 3 * 224 * 224;   // 150,528 — the shape LargeCnnComparisonBenchmark uses
    private const int OutputSize = 1000;
    private const string DefaultModel = @"C:\onnxmodels\cnn.onnx";

    private static int Main(string[] args)
    {
        var modelPath = args.Length > 0
            ? args[0]
            : Environment.GetEnvironmentVariable(OverfitEnvironment.CnnOnnx) ?? DefaultModel;

        var iterations = args.Length > 1 ? int.Parse(args[1], CultureInfo.InvariantCulture) : 30;
        var warmup = args.Length > 2 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 10;

        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"model not found: {modelPath}");

            return 2;
        }

        // The flag is read once at static init, so setting it from inside this process would be too late.
        // Say so plainly rather than reporting an empty result: an arm whose lever is not live runs
        // identical to the one it was meant to differ from, and that has happened here before.
        if (!OverfitParallel.MeasureOccupancy)
        {
            Console.Error.WriteLine(
                $"{OverfitEnvironment.ParallelOccupancy}=1 must be set in the ENVIRONMENT before this "
                + "process starts. It is read once at static init, so setting it here would do nothing.");

            return 3;
        }

        // Every lever that changes the numbers, echoed. A run whose configuration is not fully visible
        // cannot be compared with another run — reporting only one of two pool sizes is how two readings
        // six-fold apart once looked like the same configuration.
        Console.WriteLine($"model            : {modelPath}");
        Console.WriteLine($"workers          : {OverfitParallel.WorkerCount}");
        Console.WriteLine($"ProcessorCount   : {Environment.ProcessorCount}");
        Console.WriteLine($"iterations       : {iterations} measured, {warmup} warmup");

        foreach (var name in new[]
                 {
                     OverfitEnvironment.ParallelOccupancy,
                     OverfitEnvironment.ParallelChunkFactor,
                     OverfitEnvironment.ParallelRegionMajor,
                     OverfitEnvironment.ParallelWorkers,
                 })
        {
            Console.WriteLine($"{name,-32} = {Environment.GetEnvironmentVariable(name) ?? "<unset>"}");
        }

        var input = new float[InputSize];
        var output = new float[OutputSize];
        var rng = new Random(1234);

        for (var i = 0; i < InputSize; i++)
        {
            input[i] = (float)rng.NextDouble();
        }

        var model = OnnxGraphImporter.Load(modelPath, InputSize, OutputSize);
        model.Eval();
        using var backend = new OnnxGraphInferenceBackend(model);
        var engine = InferenceEngine.FromBackend(backend);

        for (var i = 0; i < warmup; i++)
        {
            engine.Run(input, output);
        }

        // A degenerate output means the model did not really run, and occupancy for a run that computed
        // nothing is worse than no reading at all.
        var maxAbs = 0f;

        for (var i = 0; i < OutputSize; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(output[i]));
        }

        if (maxAbs == 0f)
        {
            Console.Error.WriteLine("output is all zeros after warmup — the model did not run. Refusing to report.");

            return 4;
        }

        // Reset AFTER the warmup, so JIT, first-touch page faults and the pool's wake-up are not counted.
        OverfitParallel.ResetOccupancy();

        var times = new double[iterations];

        for (var i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            engine.Run(input, output);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        var report = OverfitParallel.OccupancyReport();

        Array.Sort(times);

        Console.WriteLine();
        Console.WriteLine($"wall per iteration: min {times[0]:F2} ms, median {times[iterations / 2]:F2} ms, "
                          + $"max {times[iterations - 1]:F2} ms   (NOT a benchmark result)");
        Console.WriteLine($"output max|x|     : {maxAbs:E3}");
        Console.WriteLine();
        Console.WriteLine("OCCUPANCY " + report);
        Console.WriteLine();
        Console.WriteLine(OverfitParallel.OccupancyHistogram());

        // The prose forms OccupancyReport returns when it has nothing. Treat them as failures, not results.
        if (report.StartsWith("(", StringComparison.Ordinal))
        {
            Console.Error.WriteLine("no occupancy data was recorded — this run measured nothing.");

            return 5;
        }

        return 0;
    }
}
