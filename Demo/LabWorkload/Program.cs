// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.InteropServices;
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
//   POST /fault/clear            back to a good replica, no restart — cancels a running OOM allocation too
//   POST /fault/oom              allocate NATIVE memory, every page touched, until the limit kills it
//   POST /fault/crash            exit(1) — a restart without an OOM
//
// Every fault is off by default, so an unconfigured pod is a healthy replica.

var faults = new FaultState(FaultProfile.FromEnvironment());

// Constructed before the host, because MeterListener only sees instruments published after it starts and
// the hosting layer creates its own during startup.
using var runtimeSignals = new RuntimeSignalListener();

var metrics = new WorkloadMetrics(faults.Role, runtimeSignals);

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
//
// MEASURED 2026-08-08 (AN-D6): the previous version could not produce an OOM at all, and returned 200 while
// failing. It allocated managed byte[64 MB] and touched chunk[0] and chunk[^1] only. Two things went wrong
// and either alone was fatal:
//
//   1. Two writes fault in two 4 KB pages. A cgroup limit counts RESIDENT memory, not reservation, so after
//      the process reached VmSize 6.65 GB the cgroup was charged 37 MB of its 512 MiB limit and the kernel
//      OOM killer was never anywhere near it. This is the same trap the leak timer above documents in
//      detail — the fix landed there and not here.
//   2. Managed allocation cannot reach the limit even when every page IS touched, because the
//      container-aware GC sets a heap hard limit at 75% of the cgroup limit (384 MiB of 512 MiB) and throws
//      a managed OutOfMemoryException first. Thrown inside a detached Task.Run with no continuation, it was
//      swallowed: the loop stopped, nothing was logged, and GET /fault still said "healthy".
//
// So the allocation is now NATIVE — outside the GC heap, therefore not subject to its hard limit — and every
// page is written, so the cgroup is charged what was allocated. Marshal rather than NativeMemory keeps the
// project free of unsafe blocks.
app.MapPost("/fault/oom", () =>
{
    // 32 MiB per step reaches a 512 MiB limit in sixteen steps; at 100 ms that is under two seconds, which is
    // deliberate. The memory RAMP is the leak fault's job and it has a rate knob for it. What this endpoint
    // owes the detector is the EVENT.
    const int chunkBytes = 32 * 1024 * 1024;
    const int pageBytes = 4096;

    // A bound, not `while (true)`: 4 GiB is eight times the limit this lab sets, so reaching it means no kill
    // happened. That is a defect to report loudly — an injector whose failure looks like success is exactly
    // how AN-D6 survived, and it would have been read as evidence that the detector is blind.
    const long maxBytes = 4L * 1024 * 1024 * 1024;

    var token = faults.BeginOomAllocation();
    var logger = app.Logger;

    _ = Task.Run(() =>
    {
        var held = new List<IntPtr>((int)(maxBytes / chunkBytes));
        var allocated = 0L;

        try
        {
            Thread.Sleep(500);

            while (allocated < maxBytes && !token.IsCancellationRequested)
            {
                var block = Marshal.AllocHGlobal(chunkBytes);

                held.Add(block);

                for (var offset = 0; offset < chunkBytes; offset += pageBytes)
                {
                    Marshal.WriteByte(block, offset, 1);
                }

                allocated += chunkBytes;
                Thread.Sleep(100);
            }

            if (token.IsCancellationRequested)
            {
                logger.LogInformation(
                    "oom fault cancelled after {Mib} MiB — released, no kill", allocated / (1024 * 1024));

                return;
            }

            logger.LogError(
                "oom fault reached its {Mib} MiB bound WITHOUT being killed — this pod has no enforced "
                + "memory limit, so no container_oom_events_total was produced and the silence that "
                + "follows is NOT evidence about the detector", maxBytes / (1024 * 1024));
        }
        catch (Exception ex)
        {
            // Observed, not swallowed. The previous version's throw vanished and the endpoint stayed silent.
            logger.LogError(ex, "oom fault failed after {Mib} MiB", allocated / (1024 * 1024));
        }
        finally
        {
            // Only reached when the kill did NOT happen; a killed process frees nothing. Releasing here is
            // what makes POST /fault/clear a real undo rather than a label change.
            foreach (var block in held)
            {
                Marshal.FreeHGlobal(block);
            }
        }
    }, token);

    return Results.Text("allocating native memory until the container limit kills this pod");
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
