// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Net.Http.Json;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// <b>SUPERSEDED by Demo/LabLoadDriver, which runs inside the cluster.</b> Kept because it is
    /// still the only way to drive the inference-server lab, but do not reach for it by default: it
    /// depends on one <c>kubectl port-forward</c> per replica, and that path produced three separate
    /// classes of silent failure — stale forwards answering for the WRONG pod, forwards dropping
    /// mid-run and leaving a replica idle inside a peer group, and this being an xUnit test whose
    /// output is buffered until after the run it would have explained. Four lab runs were lost to
    /// those before the driver moved into the cluster, where no forward exists to go stale.
    ///
    /// <para><b>Its failure mode on a box with no forwards is <i>"no request succeeded — the lab is
    /// reachable but not serving"</i></b>, measured 2026-08-07 in the first release-gate run. That message
    /// names a symptom and not the cause, and the cause is almost always that
    /// <c>k8s\overfit\forward-replicas.cmd</c> is not running. Note this is a <b>different</b> route from
    /// the Prometheus forward in <c>k8s\monitoring\forward.cmd</c>: this test talks to the replicas, not
    /// to Prometheus, so having one up says nothing about the other.</para>
    ///
    /// Drives traffic at the Overfit replicas running in the local Kubernetes lab (<c>k8s/</c>), so the
    /// anomaly detectors have something other than flat lines to look at.
    ///
    /// <para><b>Why a test and not a console app.</b> It needs the solution's HTTP shapes, it belongs next to
    /// the code it exercises, and xUnit already provides the runner, the output plumbing and the skip
    /// mechanism. It is <c>[LongFact]</c>, so <c>dotnet test</c> never runs it by accident; set
    /// <c>OVERFIT_RUN_LONG=1</c> to run it. (This paragraph used to say it was <c>[Fact]</c> and that you
    /// should "flip it to <c>[Fact]</c>" — advice that contradicted itself and predated the environment
    /// switch added on 2026-08-06.)</para>
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

        [LabFact(LabEndpoint.Replicas)]
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

            const int ProbeAttempts = 5;
            var probeRetryDelay = TimeSpan.FromSeconds(3);

            using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };

            // The probe reports; it does NOT decide. Every endpoint given is driven.
            //
            // <b>Excluding an endpoint on a failed probe cost four lab runs and this is the fix.</b> The probe
            // was a one-shot /health at start-up, and a kubectl port-forward establishes its tunnel lazily and
            // refuses connections often enough that a healthy pod fails it by chance. Retrying five times did
            // not stop it either — the last run dropped the degraded replica again, minutes before that same
            // pod answered /health in 6 ms and a chat request in 562 ms.
            //
            // The damage is out of all proportion to the cause. A dropped endpoint is not a smaller load: it
            // is an IDLE REPLICA INSIDE A PEER GROUP for the whole run, which every detector downstream reads
            // as a fabricated outlier and a starved peer. When the one dropped is the deliberately degraded
            // replica, the experiment is asking whether the guard can find a fault on a pod that received no
            // traffic at all — and "it found nothing" then means nothing.
            //
            // A transient failure must therefore cost a few failed requests, not a peer. Genuinely dead
            // endpoints are caught where they should be: by per-pod request rates read from Prometheus before
            // anything is measured.
            var reachable = new List<string>(endpoints);
            var unproven = new List<string>();

            foreach (var endpoint in endpoints)
            {
                var answered = false;

                for (var attempt = 1; attempt <= ProbeAttempts && !answered; attempt++)
                {
                    try
                    {
                        using var probe = await client.GetAsync($"{endpoint}/health");
                        answered = probe.IsSuccessStatusCode;
                    }
                    catch (Exception ex)
                    {
                        _out.WriteLine($"  {endpoint}: {ex.GetType().Name} (attempt {attempt})");
                    }

                    if (!answered && attempt < ProbeAttempts)
                    {
                        await Task.Delay(probeRetryDelay);
                    }
                }

                if (!answered)
                {
                    unproven.Add(endpoint);
                }
            }

            if (unproven.Count > 0)
            {
                _out.WriteLine("");
                _out.WriteLine($"  {unproven.Count} endpoint(s) never answered /health and are being driven "
                               + "anyway: " + string.Join(", ", unproven));
                _out.WriteLine("  If they are genuinely down their requests will fail and show in the table "
                               + "below, which is recoverable. Dropping them would not have been.");
                _out.WriteLine("");
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
