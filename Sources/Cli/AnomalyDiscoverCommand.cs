// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>
    /// Looks at a cluster's Prometheus and proposes a guard configuration for it.
    ///
    /// <para><b>The first thing to run at a new site, before anything is deployed.</b> Every channel the
    /// guard cannot bind is one it reports blind on for ever, and blindness is indistinguishable from health
    /// at every layer below — so "this guard will not see these four things" is a fact an operator should be
    /// handed at install time rather than discover from a month of silence.</para>
    ///
    /// <para><b>It exists because hand-authoring the mapping does not work.</b> The mapping for this
    /// project's own lab was written by the author of the system and still left two channels of thirteen
    /// unbound. If that is the base rate for someone who knows the code, a customer will do worse.</para>
    ///
    /// <para><b>It proposes; it does not decide.</b> A channel with two evidenced candidates is reported as
    /// ambiguous and left out of the generated file. The clearest case is the error rate: whether a 4xx counts
    /// as an error is a business question, not a property of any series.</para>
    /// </summary>
    internal static class AnomalyDiscoverCommand
    {
        public static async Task<int> RunAsync(
            string prometheus, string namespaceName, string podRegex, string? outPath, CancellationToken ct)
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };

            HashSet<string> names;

            try
            {
                names = await ReadNamesAsync(http, prometheus, ct).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException)
            {
                Console.Error.WriteLine($"Could not read the metric index from {prometheus}: {ex.Message}");

                return 1;
            }

            Console.WriteLine($"Prometheus  {prometheus}");
            Console.WriteLine($"scope       namespace {namespaceName}, pods /{podRegex}/");
            Console.WriteLine($"metric names known to this Prometheus: {names.Count}");

            var stacks = MetricDiscovery.Stacks(names);

            Console.WriteLine(stacks.Count == 0
                ? "runtime families: none recognised — only container-level channels will bind"
                : $"runtime families present in the cluster: {string.Join(", ", stacks)}");
            Console.WriteLine();

            // Evidence is per series and costs one instant query each, so it is cached: several channels
            // share a candidate (the request counter feeds both throughput and errors).
            var evidence = new Dictionary<string, int>(StringComparer.Ordinal);

            var discovered = MetricDiscovery.Propose(
                names,
                series => Count(http, prometheus, series, namespaceName, podRegex, evidence, ct));

            Report(discovered);

            if (string.IsNullOrWhiteSpace(outPath))
            {
                Console.WriteLine();
                Console.WriteLine("Pass --out to write a configuration draft containing the resolved channels.");

                return 0;
            }

            await File.WriteAllTextAsync(outPath, Draft(prometheus, namespaceName, podRegex, discovered), ct)
                .ConfigureAwait(false);

            Console.WriteLine();
            Console.WriteLine($"Draft written to {outPath}. Review it — ambiguous channels were deliberately");
            Console.WriteLine("left out, and thresholds are not in it: those come from a shadow period, not a guess.");

            return 0;
        }

        private static void Report(IReadOnlyList<ChannelDiscovery> discovered)
        {
            Console.WriteLine($"   {"channel",-24}{"status",-12}{"series",-46}{"pods",6}");

            var blind = 0;
            var ambiguous = 0;

            for (var i = 0; i < discovered.Count; i++)
            {
                var entry = discovered[i];

                switch (entry.Outcome)
                {
                    case DiscoveryOutcome.Resolved:
                        Console.WriteLine($"   {entry.Metric,-24}{"bound",-12}{entry.Chosen.Source,-46}"
                                          + $"{entry.Chosen.PodsReporting,6}");
                        break;

                    case DiscoveryOutcome.Ambiguous:
                        ambiguous++;
                        Console.WriteLine($"   {entry.Metric,-24}{"AMBIGUOUS",-12}{"choose one:",-46}");

                        for (var c = 0; c < entry.Candidates.Count; c++)
                        {
                            if (entry.Candidates[c].IsEvidenced)
                            {
                                Console.WriteLine($"   {"",-36}{entry.Candidates[c].Source,-46}"
                                                  + $"{entry.Candidates[c].PodsReporting,6}");
                            }
                        }

                        break;

                    default:
                        blind++;
                        Console.WriteLine($"   {entry.Metric,-24}{"BLIND",-12}"
                                          + $"{Rejected(entry.Candidates),-46}");
                        break;
                }
            }

            Console.WriteLine();
            Console.WriteLine($"{discovered.Count - blind - ambiguous} of {discovered.Count} channels bound, "
                              + $"{ambiguous} need a decision, {blind} will report blind.");
            Console.WriteLine("A blind channel produces no findings, which looks exactly like health — this list");
            Console.WriteLine("is the part worth reading before the guard is deployed, not after.");
        }

        private static string Rejected(IReadOnlyList<MetricCandidate> candidates)
        {
            if (candidates.Count == 0)
            {
                return "no known series for this channel";
            }

            // Present in the cluster and not exported by these pods — a different problem from "absent", and
            // one an operator can often fix by turning an exporter on.
            return $"{candidates[0].Source} exists, these pods do not export it";
        }

        private static string Draft(
            string prometheus, string namespaceName, string podRegex,
            IReadOnlyList<ChannelDiscovery> discovered)
        {
            var text = new StringBuilder();

            text.Append("{\n");
            text.Append("  \"prometheus\": \"").Append(prometheus).Append("\",\n");
            text.Append("  \"namespace\": \"").Append(namespaceName).Append("\",\n");
            text.Append("  \"podRegex\": \"").Append(podRegex).Append("\",\n");
            text.Append(MetricDiscovery.ToConfigJson(discovered));
            text.Append("\n}\n");

            return text.ToString();
        }

        private static async Task<HashSet<string>> ReadNamesAsync(
            HttpClient http, string prometheus, CancellationToken ct)
        {
            var url = $"{prometheus.TrimEnd('/')}/api/v1/label/__name__/values";

            using var response = await http.GetAsync(url, ct).ConfigureAwait(false);

            response.EnsureSuccessStatusCode();

            await using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            using var json = await JsonDocument.ParseAsync(stream, cancellationToken: ct).ConfigureAwait(false);

            var names = new HashSet<string>(StringComparer.Ordinal);

            foreach (var element in json.RootElement.GetProperty("data").EnumerateArray())
            {
                var name = element.GetString();

                if (name is not null)
                {
                    names.Add(name);
                }
            }

            return names;
        }

        /// <summary>
        /// How many of the selected pods export <paramref name="series"/>, right now.
        ///
        /// <para>Synchronous by necessity — <see cref="MetricDiscovery.Propose"/> takes a plain function so
        /// that its decisions stay testable without a cluster, and the alternative would be dragging async
        /// through the pure core to save a few seconds in a command a human runs once.</para>
        /// </summary>
        private static int Count(
            HttpClient http, string prometheus, string series, string namespaceName, string podRegex,
            Dictionary<string, int> cache, CancellationToken ct)
        {
            if (cache.TryGetValue(series, out var cached))
            {
                return cached;
            }

            var query = $"count by (pod) ({series}{{namespace=\"{namespaceName}\",pod=~\"{podRegex}\"}})";
            var url = $"{prometheus.TrimEnd('/')}/api/v1/query?query={Uri.EscapeDataString(query)}";
            var pods = 0;

            try
            {
                using var response = http.GetAsync(url, ct).GetAwaiter().GetResult();

                response.EnsureSuccessStatusCode();

                var body = response.Content.ReadAsStringAsync(ct).GetAwaiter().GetResult();

                using var json = JsonDocument.Parse(body);

                pods = json.RootElement.GetProperty("data").GetProperty("result").GetArrayLength();
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException)
            {
                // Counted as "no evidence", and said out loud: a query that failed must not be reported as a
                // metric these pods do not export, because the fixes are different.
                Console.Error.WriteLine(
                    $"   (could not verify {series}: {ex.Message}) — treated as not exported");
            }

            cache[series] = pods;

            return pods;
        }

        private static string Format(double value)
            => value.ToString("G4", CultureInfo.InvariantCulture);
    }
}
