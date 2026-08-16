// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;

// Drives traffic at the lab workload on a daily curve, from inside the cluster.
//
// ── Why it runs as a pod rather than as a test on the workstation ────────────────────────────────────────
// Every previous load run went through one `kubectl port-forward` per replica, and that path produced three
// separate classes of silent failure: stale forwards from an earlier run kept their ports and answered for
// the WRONG pod, so traffic went somewhere other than intended and nothing said so; a forward that dropped
// mid-run left a replica idle inside a peer group, which every detector reads as a fabricated outlier; and
// the runs were driven from an xUnit test whose output xUnit buffers until the test ends, so the diagnostic
// naming the dropped endpoint only appeared after the run it would have explained.
//
// Inside the cluster there is no forward to go stale. The headless Service resolves to every ready pod IP,
// and re-resolving picks up scaling without anyone maintaining a list.
//
// ── Why the rate is paced rather than the concurrency fixed ──────────────────────────────────────────────
// With N workers each waiting for a response, the offered rate is N / latency — so a replica that slows down
// receives LESS traffic, and one that stalls receives almost none. That is backwards for a fault experiment:
// the throttled replica in the old lab read as "idle" and a pre-flight check aborted the run over it. Pacing
// by requests per second decouples offered load from service time, which is what a real client does.
//
// Configuration, all optional:
//   LAB_TARGET       headless service DNS      default lab-workload.lab.svc.cluster.local
//   LAB_PORT         workload port             default 8080
//   LAB_PEAK_RPS     requests/second at peak   default 24
//   LAB_TROUGH       fraction of peak at 3am   default 0.25
//   LAB_PERIOD_MIN   length of one "day"       default 1440 (a real day)
//   LAB_JITTER       random walk on the curve  default 0.10

var target = Env("LAB_TARGET", "lab-workload.lab.svc.cluster.local");
var port = (int)Number("LAB_PORT", 8080);
var peakRps = Number("LAB_PEAK_RPS", 24.0);
var trough = Math.Clamp(Number("LAB_TROUGH", 0.25), 0.0, 1.0);
var periodMinutes = Number("LAB_PERIOD_MIN", 1440.0);
var jitter = Math.Clamp(Number("LAB_JITTER", 0.10), 0.0, 1.0);

Console.WriteLine($"lab-load-driver: target={target}:{port} peak={peakRps:F1} rps trough={trough:P0} "
                  + $"period={periodMinutes:F0} min jitter={jitter:P0}");

using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };

var endpoints = Array.Empty<string>();
var resolvedAt = DateTimeOffset.MinValue;
var sent = 0L;
var failed = 0L;
var started = Stopwatch.GetTimestamp();

// A drift term rather than per-request noise: real traffic wanders around its curve over minutes, and
// independent per-request jitter would average out over any window the detectors look at, leaving a shape
// far smoother than anything a cluster actually sees.
var drift = 1.0;
var random = new Random();

using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cancellation.Cancel();
};

var reportedAt = DateTimeOffset.UtcNow;

// #pragma BOUND: exits on SIGINT/SIGTERM through the cancellation token, which is how a pod is stopped.
while (!cancellation.IsCancellationRequested)
{
    var now = DateTimeOffset.UtcNow;

    if (now - resolvedAt > TimeSpan.FromSeconds(30))
    {
        endpoints = await ResolveAsync(target, port).ConfigureAwait(false);
        resolvedAt = now;
    }

    if (endpoints.Length == 0)
    {
        Console.WriteLine($"{now:HH:mm:ss} no endpoints for {target}; retrying");
        await Task.Delay(TimeSpan.FromSeconds(5), cancellation.Token).ConfigureAwait(false);

        continue;
    }

    // The daily curve. Trough at the period's start, peak halfway — the absolute phase does not matter for
    // any detector, only that the slope is real and lasts hours.
    var elapsedMinutes = Stopwatch.GetElapsedTime(started).TotalMinutes;
    var phase = elapsedMinutes / periodMinutes % 1.0;
    var shape = trough + ((1.0 - trough) * (0.5 - (0.5 * Math.Cos(2.0 * Math.PI * phase))));

    drift = Math.Clamp(drift + ((random.NextDouble() - 0.5) * jitter * 0.1), 1.0 - jitter, 1.0 + jitter);

    var rps = Math.Max(0.5, peakRps * shape * drift);
    var interval = TimeSpan.FromSeconds(1.0 / rps);

    var endpoint = endpoints[(int)(Interlocked.Read(ref sent) % endpoints.Length)];

    // Fire and forget, so a slow or stalling replica cannot hold the pacer back. That is the whole point of
    // pacing: offered load is a property of the client, not of how the server happens to be feeling.
    //
    // OVERFIT046 — WHAT MAKES THE DISCARD SAFE: SendAsync's whole body is inside a try/catch(Exception) that
    // counts the failure into `failed`, and the one statement before that try is an Interlocked.Increment
    // that cannot throw. The response is disposed inside the same try, so a fault during disposal is caught
    // too. The task therefore never completes faulted and there is nothing for an observer to observe.
    // Awaiting it here would restore exactly the coupling the pacer exists to remove, and the header comment
    // above records what that coupling cost: a throttled replica received less traffic and read as idle.
    //
    // WHAT IS LOST: the failure is COUNTED, not attributed. No URL, no status code and no exception type
    // reaches the log, so `failed=N` in the five-minute report is the only evidence, and a cancellation
    // during shutdown is counted the same as a refused connection. That is acceptable here because the
    // driver's job is offered load, and the workload's own metrics are what the guard reads.
#pragma warning disable OVERFIT046
    _ = SendAsync(client, endpoint);
#pragma warning restore OVERFIT046

    if (now - reportedAt > TimeSpan.FromMinutes(5))
    {
        reportedAt = now;
        Console.WriteLine(
            $"{now:HH:mm:ss} phase={phase:P0} rps={rps:F1} endpoints={endpoints.Length} "
            + $"sent={Interlocked.Read(ref sent)} failed={Interlocked.Read(ref failed)}");
    }

    await Task.Delay(interval, cancellation.Token).ConfigureAwait(false);
}

Console.WriteLine($"stopped; sent={sent} failed={failed}");

return 0;

async Task SendAsync(HttpClient http, string url)
{
    Interlocked.Increment(ref sent);

    try
    {
        using var response = await http.PostAsync(url, content: null).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            Interlocked.Increment(ref failed);
        }
    }
    catch (Exception)
    {
        Interlocked.Increment(ref failed);
    }
}

static async Task<string[]> ResolveAsync(string host, int port)
{
    try
    {
        var addresses = await Dns.GetHostAddressesAsync(host).ConfigureAwait(false);
        var urls = new List<string>(addresses.Length);

        foreach (var address in addresses)
        {
            if (address.AddressFamily == AddressFamily.InterNetwork)
            {
                urls.Add($"http://{address}:{port}/work");
            }
        }

        urls.Sort(StringComparer.Ordinal);

        return urls.ToArray();
    }
    catch (SocketException)
    {
        return [];
    }
}

static string Env(string name, string fallback)
    => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : fallback;

static double Number(string name, double fallback)
    => double.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Float,
        CultureInfo.InvariantCulture, out var value) && value > 0.0
        ? value
        : fallback;
