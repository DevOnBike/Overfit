// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LabWorkload;

// A stand-in for a client's application: it serves requests, exports metrics, and can be told to misbehave
// in a way somebody chose. See FaultProfile for why the lab stopped using the real inference server for this,
// and FaultState for why the faults are switchable at runtime rather than only at deploy time.
//
//   GET  /health                 readiness
//   GET  /metrics                Prometheus exposition
//   POST /work                   one unit of simulated work
//
//   GET  /fault                  what this replica is currently pretending to be
//   POST /fault/latency?ms=&jitter=
//   POST /fault/stall?probability=&seconds=
//   POST /fault/errors?rate=
//   POST /fault/leak?bytesPerSecond=
//   POST /fault/cpu?msPerRequest=
//   POST /fault/clear            back to a good replica, no restart
//   POST /fault/oom              allocate until the container limit kills it
//   POST /fault/crash            exit(1) — a restart without an OOM
//
// Every fault is off by default, so an unconfigured pod is a healthy replica.

var faults = new FaultState(FaultProfile.FromEnvironment());
var metrics = new WorkloadMetrics(faults.Role);

// Retained deliberately and never released — a leak with a rate somebody chose, which is the one fault the
// real lab could not produce and the one the trend family exists to catch.
var leaked = new List<byte[]>();
var leakLock = new object();

var builder = WebApplication.CreateSlimBuilder(args);
builder.Logging.AddSimpleConsole(o => o.TimestampFormat = "HH:mm:ss ");

var app = builder.Build();

app.Logger.LogInformation("lab-workload up: role={Role} — {Faults}", faults.Role, faults.Describe());

// A timer rather than per-request, so the leak rate is what it says regardless of traffic. Tying it to
// requests would make the leak a function of load, and a trend finding on memory could then never be
// separated from a trend in traffic.
using var leakTimer = new Timer(
    _ =>
    {
        var perSecond = faults.LeakBytesPerSecond;

        if (perSecond <= 0.0)
        {
            return;
        }

        var chunk = new byte[(int)Math.Clamp(perSecond, 1, 64 * 1024 * 1024)];

        // EVERY page, not just the ends. The previous version touched chunk[0] and chunk[^1] and the comment
        // above it named the exact failure it was trying to prevent — but two writes fault in two 4 KB
        // pages, so a 2 MB chunk moved the working set by 8 KB. Anything past ~85 KB is a Large Object Heap
        // allocation served from pre-zeroed OS pages, which the runtime need not write to, so the pages stay
        // mapped to the shared zero page and never become resident.
        //
        // Measured before the fix, at 2 MB/s for four minutes: RSS 39 -> 36 -> 37 -> 24 MiB, i.e. falling,
        // while dotnet_gc_heap_size_bytes rose 542% of typical and the guard raised a GcGen2HeapBytes
        // incident. So the leak fault has only ever exercised the HEAP channel, and every statement that it
        // tests MemoryWorkingSetBytes — including the trend family's headline case — was untested.
        for (var offset = 0; offset < chunk.Length; offset += 4096)
        {
            chunk[offset] = 1;
        }

        lock (leakLock)
        {
            leaked.Add(chunk);
        }
    },
    null,
    TimeSpan.FromSeconds(1),
    TimeSpan.FromSeconds(1));

app.MapGet("/health", () => Results.Text("ok"));

app.MapGet("/metrics", () =>
    Results.Text(metrics.Render(), "text/plain; version=0.0.4; charset=utf-8"));

app.MapGet("/fault", () => Results.Text(faults.Describe()));

app.MapPost("/work", async () =>
{
    var started = Stopwatch.GetTimestamp();
    var random = Random.Shared;

    // Multiplicative scatter around the median, then an ADDITIVE stall on top. The two are modelled
    // separately because they move the quantiles differently: scaling lifts every quantile by the same
    // proportion, while a fixed wait added to whatever is in flight moves a p50 far more, relatively, than it
    // moves a p99. Reproducing that asymmetry is the reason the stall is not just a bigger jitter.
    var seconds = faults.LatencyMs / 1000.0
                  * (1.0 + ((random.NextDouble() - 0.5) * faults.LatencyJitter));

    if (faults.StallProbability > 0.0 && random.NextDouble() < faults.StallProbability)
    {
        seconds += faults.StallSeconds;
    }

    var burnMs = faults.CpuBurnMs;

    if (burnMs > 0.0)
    {
        var burnUntil = Stopwatch.GetTimestamp() + (long)(burnMs / 1000.0 * Stopwatch.Frequency);

        // #pragma BOUND: bounded by a timestamp taken before the loop; Stopwatch is monotonic.
        while (Stopwatch.GetTimestamp() < burnUntil)
        {
        }
    }

    await Task.Delay(TimeSpan.FromSeconds(Math.Max(0.0, seconds)));

    var failed = faults.ErrorRate > 0.0 && random.NextDouble() < faults.ErrorRate;

    metrics.Observe(Stopwatch.GetElapsedTime(started).TotalSeconds, failed);

    return failed ? Results.StatusCode(500) : Results.Text("done");
});

app.MapPost("/fault/latency", (double? ms, double? jitter) =>
{
    if (ms is { } value)
    {
        faults.LatencyMs = value;
    }

    if (jitter is { } scatter)
    {
        faults.LatencyJitter = scatter;
    }

    return Results.Text(faults.Describe());
});

app.MapPost("/fault/stall", (double? probability, double? seconds) =>
{
    if (probability is { } chance)
    {
        faults.StallProbability = chance;
    }

    if (seconds is { } duration)
    {
        faults.StallSeconds = duration;
    }

    return Results.Text(faults.Describe());
});

app.MapPost("/fault/errors", (double? rate) =>
{
    faults.ErrorRate = rate ?? 0.0;

    return Results.Text(faults.Describe());
});

app.MapPost("/fault/leak", (double? bytesPerSecond) =>
{
    faults.LeakBytesPerSecond = bytesPerSecond ?? 0.0;

    return Results.Text(faults.Describe());
});

app.MapPost("/fault/cpu", (double? msPerRequest) =>
{
    faults.CpuBurnMs = msPerRequest ?? 0.0;

    return Results.Text(faults.Describe());
});

app.MapPost("/fault/clear", () =>
{
    faults.Clear();

    return Results.Text(faults.Describe());
});

// Allocates until the container's memory limit kills the process. The kill is the point: it is the only way
// to produce a real container_oom_events_total and a real restart, and the peer detector is structurally
// blind to a single OOM — one event over a small share of the window scores below any usable effect size, so
// only an absolute rule catches it. That claim needs a real event to be tested against.
//
// Runs on a background thread so the response is sent before the process dies; without that the caller sees
// a connection reset and cannot tell an OOM from a network fault.
app.MapPost("/fault/oom", () =>
{
    _ = Task.Run(() =>
    {
        Thread.Sleep(500);

        var held = new List<byte[]>();

        // The bound is the container memory limit and it is enforced by the kernel, not by this process —
        // which is the whole behaviour being reproduced. An in-process counter would stop before the OOM and
        // produce no event at all. Without a limit on the pod this would take the node down instead, so the
        // manifest that exposes this endpoint is the manifest that must set one.
#pragma warning disable OVERFIT023
        while (true)
#pragma warning restore OVERFIT023
        {
            var chunk = new byte[64 * 1024 * 1024];
            chunk[0] = 1;
            chunk[^1] = 1;
            held.Add(chunk);
            Thread.Sleep(50);
        }
    });

    return Results.Text("allocating until the container limit kills this pod");
});

// A restart WITHOUT an OOM, so the two can be told apart downstream: ContainerRestarts moves, OomEventsRate
// does not. Reproducing them separately is what makes the distinction testable at all.
app.MapPost("/fault/crash", () =>
{
    _ = Task.Run(() =>
    {
        Thread.Sleep(500);
        Environment.Exit(1);
    });

    return Results.Text("exiting in 500ms");
});

app.Run();
