using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Runtime;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

// One workload, one process, no BenchmarkDotNet child processes — a profiler must attribute samples to
// the code under study, and BenchmarkDotNet's host/child split scatters them across two binaries.
//
// argv[0] = model path, argv[1] = seconds of steady-state work after warmup.
// PROF_NODES=1 prints the per-layer table. DOTNET_PROCESSOR_COUNT sets the thread count and is READ BACK
// out of the process before any reading is used - a ladder whose rungs all ran at 32 threads would look
// like a scaling wall and would be an instrument fault.
//
// It lives under Scripts/ rather than artifacts/ because .gitignore excludes artifacts/ wholesale, and
// this repository has already lost a helper that way once (see CLAUDE.md on Scripts/lab.py).
//
//   dotnet build Scripts/ProfHarness/ProfHarness.csproj -c Release
// PROF_AFFINITY is a hex core mask, applied before anything reads the processor count.
// PROF_ENGINE=ort runs ONNX Runtime through the same loop, with PROF_ORT_THREADS as its intra-op count.
//
// Build:  dotnet build Scripts/ProfHarness/ProfHarness.csproj -c Release
// Run:    bin/Release/net10.0/ProfHarness.exe <model path> <seconds>
//
// It found the result in docs/measured-baselines.md that BenchmarkDotNet could not: with the pool sized
// to real cores, VGG-16 scales 6.93x across 16 cores against the machine's measured 14.30x, so the
// remaining gap to ONNX Runtime is parallel scaling, not the micro-kernel.
internal static class Prof
{
    private const int InputSize = 3 * 224 * 224;
    private const int OutputSize = 1000;

    private static int Main(string[] args)
    {
        // Affinity FIRST, before anything reads Environment.ProcessorCount — .NET caches it on first
        // access, so a mask applied later would leave the pool sized for the whole machine while the
        // threads ran on a subset. The count is printed below so this is checked, not assumed: the first
        // attempt at this probe silently failed to apply the mask, and six readings that were all the same
        // arm looked exactly like six arms with no difference between them.
        var affinity = Environment.GetEnvironmentVariable("PROF_AFFINITY");

        if (!string.IsNullOrEmpty(affinity))
        {
            Process.GetCurrentProcess().ProcessorAffinity = (nint)Convert.ToInt64(affinity, 16);
        }

        var path = args.Length > 0 ? args[0] : @"C:\onnxmodels\vgg16.onnx";
        var seconds = args.Length > 1 ? double.Parse(args[1]) : 20.0;

        var input = new float[InputSize];
        var output = new float[OutputSize];
        var rng = new Random(1234);
        for (var i = 0; i < InputSize; i++) { input[i] = (float)rng.NextDouble(); }

        // PROF_ENGINE=ort runs ONNX Runtime through the same loop, the same affinity mask and the same
        // warmup, so its scaling curve can be put beside ours. Its thread count is set explicitly rather
        // than left to its own default, because "our 16 workers against their 32" is not a comparison.
        if (Environment.GetEnvironmentVariable("PROF_ENGINE") == "ort")
        {
            return RunOnnxRuntime(path, input, seconds);
        }

        // PROF_ENGINE=pool measures OverfitParallel itself on a perfectly balanced, cache-resident,
        // register-only workload. Convolution scales 7.08x across 16 cores where ONNX Runtime's scales
        // 12.12x; this separates "the pool cannot fan out" from "the convolution's work decomposition is
        // wrong", and those call for opposite work.
        if (Environment.GetEnvironmentVariable("PROF_ENGINE") == "pool")
        {
            return RunPoolScaling(seconds);
        }

        using var model = OnnxGraphImporter.Load(path, InputSize, OutputSize);
        model.Eval();
        using var engine = InferenceEngine.FromBackend(new OnnxGraphInferenceBackend(model));

        // Warmup long enough for tier-1 promotion. Measured on this box: 5 warmup calls left VGG-16 running
        // tier-0 code and reading 5.929 ms against a steady-state 1.482 ms.
        for (var i = 0; i < 40; i++) { engine.Run(input, output); }

        Console.WriteLine($"[PROF] ProcessorCount={Environment.ProcessorCount}");
        OnnxGraphModel.ProfileNodes = Environment.GetEnvironmentVariable("PROF_NODES") == "1";
        model.ResetNodeProfile();
        OverfitParallel.ResetOccupancy();
        Console.WriteLine("[PROF] warmup done, steady state starts now");
        Console.Out.Flush();

        var sw = Stopwatch.StartNew();
        var calls = 0;
        while (sw.Elapsed.TotalSeconds < seconds) { engine.Run(input, output); calls++; }
        sw.Stop();

        Console.WriteLine($"[PROF] {calls} calls in {sw.Elapsed.TotalSeconds:F2} s = "
                          + $"{sw.Elapsed.TotalMilliseconds / calls:F3} ms/call");
        Console.WriteLine($"[PROF] checksum {output[0]:E3}");
        if (OnnxGraphModel.ProfileNodes) { Console.WriteLine(model.PerNodeProfileReport()); }

        Console.WriteLine($"[PROF] parallel: {OverfitParallel.OccupancyReport()}");
        return 0;
    }

    /// <summary>
    /// ONNX Runtime through the identical loop, so the two scaling curves are comparable.
    ///
    /// <para><c>PROF_ORT_THREADS</c> sets both intra- and inter-op counts. It is required rather than
    /// defaulted: ONNX Runtime picks its own count from the machine, so an unset arm would compare our
    /// masked core budget against its unmasked one and the ladder would be meaningless.</para>
    /// </summary>
    private static int RunOnnxRuntime(string path, float[] input, double seconds)
    {
        var threads = Environment.GetEnvironmentVariable("PROF_ORT_THREADS");

        if (string.IsNullOrEmpty(threads))
        {
            Console.Error.WriteLine("[PROF] PROF_ORT_THREADS is unset, so the arm would not be like-for-like.");
            return 2;
        }

        using var options = new SessionOptions
        {
            IntraOpNumThreads = int.Parse(threads),
            InterOpNumThreads = 1,
        };

        // PROF_ORT_OPT=extended drops ONNX Runtime below the layout-optimisation level, so its NCHWc
        // transform does not run and Conv falls back to MlasConv - im2col plus SGEMM, the same structure we
        // use. The difference between the two levels is the layout's worth measured on their own assembly,
        // with everything else held constant, and no port needed to find it out.
        var level = Environment.GetEnvironmentVariable("PROF_ORT_OPT");

        if (level == "extended")
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
        }

        if (level == "basic")
        {
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
        }

        // PROF_ORT_PROFILE=1 makes ONNX Runtime write its own per-node timings, which is the only way to put
        // their layer budget beside ours. It writes a JSON file and its path is printed on exit.
        if (Environment.GetEnvironmentVariable("PROF_ORT_PROFILE") == "1")
        {
            options.EnableProfiling = true;
            options.ProfileOutputPathPrefix = "ort_profile";
        }

        using var session = new InferenceSession(path, options);

        var name = session.InputMetadata.Keys.First();
        var dims = session.InputMetadata[name].Dimensions;
        var shape = new int[dims.Length];

        for (var i = 0; i < dims.Length; i++) { shape[i] = dims[i] <= 0 ? 1 : dims[i]; }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(input, shape)),
        };

        for (var i = 0; i < 40; i++) { using var warm = session.Run(inputs); }

        Console.WriteLine($"[PROF] ProcessorCount={Environment.ProcessorCount}");
        Console.WriteLine($"[PROF] ort threads={threads}");
        Console.WriteLine("[PROF] warmup done, steady state starts now");
        Console.Out.Flush();

        var sw = Stopwatch.StartNew();
        var calls = 0;
        var last = 0f;

        while (sw.Elapsed.TotalSeconds < seconds)
        {
            using var results = session.Run(inputs);
            last = results[0].AsTensor<float>().GetValue(0);
            calls++;
        }

        sw.Stop();

        Console.WriteLine($"[PROF] {calls} calls in {sw.Elapsed.TotalSeconds:F2} s = "
                          + $"{sw.Elapsed.TotalMilliseconds / calls:F3} ms/call");
        Console.WriteLine($"[PROF] checksum {last:E3}");

        if (Environment.GetEnvironmentVariable("PROF_ORT_PROFILE") == "1")
        {
            Console.WriteLine($"[PROF] ort profile: {session.EndProfiling()}");
        }

        return 0;
    }

    /// <summary>
    /// OverfitParallel on work that is perfectly balanced, register-resident and touches no memory: each
    /// item runs an independent FMA chain. Anything short of the machine's 14.30x here is the pool, not the
    /// workload.
    /// </summary>
    private static unsafe int RunPoolScaling(double seconds)
    {
        const int Items = 4096;

        Console.WriteLine($"[PROF] ProcessorCount={Environment.ProcessorCount}");

        var sink = new double[Items];

        fixed (double* pSink = sink)
        {
            var context = new PoolContext(pSink);

            for (var i = 0; i < 20; i++)
            {
                OverfitParallel.For(0, Items, 1, &PoolWorker, &context);
            }

            Console.WriteLine("[PROF] warmup done, steady state starts now");
            Console.Out.Flush();

            var clock = Stopwatch.StartNew();
            var calls = 0;

            while (clock.Elapsed.TotalSeconds < seconds)
            {
                OverfitParallel.For(0, Items, 1, &PoolWorker, &context);
                calls++;
            }

            clock.Stop();

            Console.WriteLine($"[PROF] {calls} calls in {clock.Elapsed.TotalSeconds:F2} s = "
                              + $"{clock.Elapsed.TotalMilliseconds / calls:F3} ms/call");
            Console.WriteLine($"[PROF] checksum {sink[0]:E3}");
        }

        return 0;
    }

    private static unsafe void PoolWorker(int start, int end, void* contextPtr)
    {
        ref readonly var context = ref System.Runtime.CompilerServices.Unsafe.AsRef<PoolContext>(contextPtr);

        for (var item = start; item < end; item++)
        {
            var a = 1.0000001;
            var b = 1.0000002;
            var c = 1.0000003;
            var d = 1.0000004;

            for (var i = 0; i < 4000; i++)
            {
                a = (a * 1.0000001) + 0.5;
                b = (b * 1.0000002) + 0.5;
                c = (c * 1.0000003) + 0.5;
                d = (d * 1.0000004) + 0.5;
            }

            context.Sink[item] = a + b + c + d;
        }
    }

    private readonly unsafe struct PoolContext
    {
        public readonly double* Sink;

        public PoolContext(double* sink)
        {
            Sink = sink;
        }
    }
}
