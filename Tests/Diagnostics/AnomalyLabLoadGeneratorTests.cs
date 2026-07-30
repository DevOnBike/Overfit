// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Net.Http.Json;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// Drives traffic at the Overfit replicas running in the local Kubernetes lab (<c>k8s/</c>), so the
    /// anomaly detectors have something other than flat lines to look at.
    ///
    /// <para><b>Why a test and not a console app.</b> It needs the solution's HTTP shapes, it belongs next to
    /// the code it exercises, and xUnit already provides the runner, the output plumbing and the skip
    /// mechanism. It is <c>[LongFact]</c>, so <c>dotnet test</c> never runs it by accident — flip it to
    /// <c>[Fact]</c> temporarily, exactly as with the other diagnostics here.</para>
    ///
    /// <para><b>Why the skew knob is the point.</b> Even load across identical replicas produces a peer group
    /// with nothing to find, which proves only that the detector does not hallucinate. The interesting case
    /// is deliberate imbalance: one replica taking several times the traffic of its siblings. That is
    /// simultaneously the fault a peer detector should catch <i>and</i> the false positive it must not raise
    /// — a busier pod legitimately uses more CPU, which is why <c>PeerSignalKind.LoadSensitive</c> demands a
    /// per-pod work metric before it will compare anything. Running this with a skew and then watching both
    /// the raw and the normalised panels is the cheapest demonstration that the distinction is real.</para>
    ///
    /// <para><b>Prerequisites.</b> The lab must be up (<c>k8s\monitoring\install.cmd</c>,
    /// <c>k8s\overfit\deploy.cmd</c>) and one port-forward per replica must be open —
    /// <c>k8s\overfit\forward-replicas.cmd</c> does that and prints the URLs this test defaults to.</para>
    ///
    /// <para>Configuration is by environment variable so the same test covers a smoke run and a long soak:
    /// <c>OVERFIT_LAB_ENDPOINTS</c> (comma-separated base URLs), <c>OVERFIT_LAB_SECONDS</c>,
    /// <c>OVERFIT_LAB_CONCURRENCY</c>, <c>OVERFIT_LAB_SKEW</c>, <c>OVERFIT_LAB_MAX_TOKENS</c>.</para>
    /// </summary>
    public sealed class AnomalyLabLoadGeneratorTests
    {
        private readonly ITestOutputHelper _out;

        public AnomalyLabLoadGeneratorTests(ITestOutputHelper output) => _out = output;

        private static string[] Endpoints()
        {
            var configured = Environment.GetEnvironmentVariable("OVERFIT_LAB_ENDPOINTS");

            if (!string.IsNullOrWhiteSpace(configured))
            {
                return configured.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            }

            // Matches k8s\overfit\forward-replicas.cmd, one local port per replica.
            return ["http://127.0.0.1:8081", "http://127.0.0.1:8082", "http://127.0.0.1:8083"];
        }

        private static int Setting(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            return int.TryParse(raw, out var value) && value > 0 ? value : fallback;
        }

        private static double SkewSetting()
        {
            var raw = Environment.GetEnvironmentVariable("OVERFIT_LAB_SKEW");

            return double.TryParse(raw, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var value) && value > 0
                ? value
                : 1.0;
        }

        [Fact]
        public async Task DriveTrafficAtTheLabReplicas()
        {
            var endpoints = Endpoints();
            var seconds = Setting("OVERFIT_LAB_SECONDS", 120);
            var concurrency = Setting("OVERFIT_LAB_CONCURRENCY", 4);
            var maxTokens = Setting("OVERFIT_LAB_MAX_TOKENS", 32);
            var skew = SkewSetting();

            _out.WriteLine($"endpoints   : {string.Join(", ", endpoints)}");
            _out.WriteLine($"duration    : {seconds}s   concurrency: {concurrency}   max_tokens: {maxTokens}");
            _out.WriteLine($"skew        : {skew:F1}x onto {endpoints[0]}"
                + (Math.Abs(skew - 1.0) < 0.01 ? "  (even — nothing for a peer detector to find)" : ""));
            _out.WriteLine("");

            using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };

            var reachable = new List<string>();
            foreach (var endpoint in endpoints)
            {
                try
                {
                    using var probe = await client.GetAsync($"{endpoint}/health");
                    if (probe.IsSuccessStatusCode)
                    {
                        reachable.Add(endpoint);
                        continue;
                    }

                    _out.WriteLine($"  {endpoint}: /health returned {(int)probe.StatusCode}");
                }
                catch (Exception ex)
                {
                    _out.WriteLine($"  {endpoint}: unreachable ({ex.GetType().Name})");
                }
            }

            if (reachable.Count == 0)
            {
                // Not an assertion failure: with no lab running there is nothing to measure, and failing
                // here would say "the code is broken" when the truth is "the cluster is not up".
                _out.WriteLine("");
                _out.WriteLine("No replica answered. Start the lab first:");
                _out.WriteLine(@"  k8s\monitoring\install.cmd");
                _out.WriteLine(@"  k8s\overfit\deploy.cmd");
                _out.WriteLine(@"  k8s\overfit\forward-replicas.cmd");
                return;
            }

            // Weight the first endpoint by `skew` so the traffic split is deliberately uneven.
            var weights = new double[reachable.Count];
            for (var i = 0; i < weights.Length; i++)
            {
                weights[i] = i == 0 ? skew : 1.0;
            }

            var totalWeight = 0.0;
            foreach (var w in weights)
            {
                totalWeight += w;
            }

            var sent = new int[reachable.Count];
            var failed = new int[reachable.Count];
            var latencyMs = new double[reachable.Count];
            var gate = new object();

            using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(seconds));
            var wall = Stopwatch.StartNew();

            var workers = new Task[concurrency];
            for (var w = 0; w < concurrency; w++)
            {
                var seed = w;
                workers[w] = Task.Run(async () =>
                {
                    var rng = new Random(20260728 + seed);

                    while (!deadline.IsCancellationRequested)
                    {
                        // Weighted pick, so the skew shows up in the per-pod series rather than in a comment.
                        var roll = rng.NextDouble() * totalWeight;
                        var index = 0;
                        for (var i = 0; i < weights.Length; i++)
                        {
                            roll -= weights[i];
                            if (roll <= 0)
                            {
                                index = i;
                                break;
                            }
                        }

                        var payload = new
                        {
                            model = "lab",
                            max_tokens = maxTokens,
                            messages = new[]
                            {
                                new { role = "user", content = $"Give one short fact number {rng.Next(10_000)}." },
                            },
                        };

                        var started = Stopwatch.GetTimestamp();
                        try
                        {
                            using var response = await client.PostAsJsonAsync(
                                $"{reachable[index]}/v1/chat/completions", payload, deadline.Token);

                            var elapsed = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
                            var succeeded = response.IsSuccessStatusCode;

                            // Both outcomes are recorded and the loop continues. An early `return` here
                            // would leave the worker — not the iteration — and the generator would send
                            // exactly one request per worker while still reporting success.
                            lock (gate)
                            {
                                if (succeeded)
                                {
                                    sent[index]++;
                                    latencyMs[index] += elapsed;
                                }

                                if (!succeeded)
                                {
                                    failed[index]++;
                                }
                            }
                        }
                        catch (OperationCanceledException)
                        {
                            return;
                        }
                        catch
                        {
                            lock (gate)
                            {
                                failed[index]++;
                            }
                        }
                    }
                });
            }

            await Task.WhenAll(workers);
            wall.Stop();

            var totalSent = 0;
            var totalFailed = 0;
            foreach (var s in sent)
            {
                totalSent += s;
            }

            foreach (var f in failed)
            {
                totalFailed += f;
            }

            _out.WriteLine("");
            _out.WriteLine($"{"endpoint",-32} {"ok",6} {"failed",7} {"share",7} {"mean ms",9}");
            for (var i = 0; i < reachable.Count; i++)
            {
                var share = totalSent > 0 ? sent[i] * 100.0 / totalSent : 0;
                var mean = sent[i] > 0 ? latencyMs[i] / sent[i] : 0;
                _out.WriteLine($"{reachable[i],-32} {sent[i],6} {failed[i],7} {share,6:F1}% {mean,9:F0}");
            }

            _out.WriteLine("");
            _out.WriteLine($"total       : {totalSent} ok, {totalFailed} failed in {wall.Elapsed.TotalSeconds:F1}s "
                + $"({totalSent / Math.Max(wall.Elapsed.TotalSeconds, 0.001):F2} req/s)");
            _out.WriteLine("");
            _out.WriteLine("Now compare, in Grafana or at http://127.0.0.1:9090/graph:");
            _out.WriteLine("  raw        rate(process_cpu_seconds_total{namespace=\"overfit\"}[5m])");
            _out.WriteLine("  normalised rate(process_cpu_seconds_total{namespace=\"overfit\"}[5m])"
                + " / rate(overfit_chat_requests_total{namespace=\"overfit\"}[5m])");
            _out.WriteLine("");
            _out.WriteLine("With a skew, the raw series separates and the normalised one should not — which is");
            _out.WriteLine("the whole reason PeerSignalKind.LoadSensitive refuses to compare without a work metric.");

            Assert.True(totalSent > 0, "no request succeeded — the lab is reachable but not serving");
        }
    }
}
