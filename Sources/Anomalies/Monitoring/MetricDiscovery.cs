// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Works out what a workload exports and proposes a mapping for it.
    ///
    /// <para><b>Why this exists at all.</b> Every channel the guard cannot bind is a channel it reports blind
    /// on for ever, and blindness is indistinguishable from health at every layer below. Until now the only
    /// way to bind them was for a human to write thirteen entries by hand — and the mapping for this
    /// project's own lab, written by the author of the system, still left <b>two of thirteen unbound</b>. If
    /// that is the base rate for someone who knows the code, asking a customer to do it is not a plan.</para>
    ///
    /// <para><b>Pure, and takes evidence rather than fetching it.</b> The caller supplies the metric names
    /// that exist and a way to ask how many pods actually export one; the decisions are made here, where they
    /// can be tested without a cluster. The distinction between those two inputs matters more than it looks:
    /// a name existing somewhere in a cluster says nothing about whether <i>these</i> pods emit it, and a
    /// binding made on a name alone produces a query that returns nothing for ever.</para>
    ///
    /// <para><b>It refuses to break ties.</b> Several evidenced candidates give
    /// <see cref="DiscoveryOutcome.Ambiguous"/> and no binding. The error rate is the clearest case: whether
    /// a 4xx is an error is a business decision, and two exporters can both be present and both be plausible.
    /// A guess there yields a guard confidently measuring the wrong thing, which is worse than one that
    /// admits it does not know.</para>
    /// </summary>
    public static class MetricDiscovery
    {
        /// <summary>What a channel that could not be bound carries, in place of a null-filled default.</summary>
        private static MetricCandidate Unbound
            => new(string.Empty, MetricSourceKind.Gauge, string.Empty, 0, double.NaN);

        /// <summary>
        /// Proposes a binding per channel.
        /// </summary>
        /// <param name="available">Every metric name the monitoring system knows, from its name index.</param>
        /// <param name="podsReporting">
        /// How many of the pods under inspection export a given series. Return 0 for "none" — that is the
        /// answer that turns a plausible name into a rejected candidate.
        /// </param>
        /// <remarks>
        /// <b>The evidence callback is asynchronous, and that is deliberate rather than fashionable.</b>
        /// It costs one Prometheus instant query per candidate series, so the only implementation that
        /// exists does network I/O. A synchronous <c>Func&lt;string, int&gt;</c> forced its caller to either
        /// block a thread on a task — which is the deadlock OVERFIT039 guards against — or do genuinely
        /// synchronous HTTP inside an async call graph, holding the caller's thread for the whole round
        /// trip. Neither is worth a signature; the callback returns a task.
        /// </remarks>
        public static async Task<IReadOnlyList<ChannelDiscovery>> ProposeAsync(
            IReadOnlySet<string> available,
            Func<string, Task<int>> podsReporting,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            ArgumentNullException.ThrowIfNull(available);
            ArgumentNullException.ThrowIfNull(podsReporting);

            var results = new List<ChannelDiscovery>((int)MetricIndex.Count);

            for (var m = 0; m < (int)MetricIndex.Count; m++)
            {
                var metric = (MetricIndex)m;
                var candidates = MetricNameCatalog.For(metric);
                var matched = new List<MetricCandidate>(candidates.Count);
                var evidenced = 0;
                var chosen = default(MetricCandidate);

                for (var i = 0; i < candidates.Count; i++)
                {
                    var candidate = candidates[i];

                    // A histogram is exposed as three series; the name index carries the buckets, so that is
                    // what has to be looked for rather than the bare family name.
                    var probe = candidate.Kind == MetricSourceKind.HistogramSeconds
                        ? candidate.Source + "_bucket"
                        : candidate.Source;

                    if (!available.Contains(probe))
                    {
                        continue;
                    }

                    var pods = await podsReporting(probe).ConfigureAwait(false);
                    var found = candidate with
                    {
                        PodsReporting = pods
                    };

                    matched.Add(found);

                    if (pods <= 0)
                    {
                        continue;
                    }

                    evidenced++;

                    if (evidenced == 1)
                    {
                        chosen = found;
                    }
                }

                // Only when nothing known matched. A name this project has seen always beats one that merely
                // ends like the right thing — see MetricNameCatalog.SuffixesFor.
                if (evidenced == 0)
                {
                    (evidenced, chosen) =
                        await BySuffixAsync(metric, available, podsReporting, matched, chosen)
                            .ConfigureAwait(false);
                }

                var outcome = evidenced switch
                {
                    0 => DiscoveryOutcome.NotFound,
                    1 => DiscoveryOutcome.Resolved,
                    _ => DiscoveryOutcome.Ambiguous,
                };

                results.Add(new ChannelDiscovery(
                    metric,
                    outcome,

                    // Empty strings rather than default(MetricCandidate), whose string fields are null. A
                    // caller reading Chosen.Source on an unresolved channel is doing something reasonable
                    // and must not get a null reference for it.
                    outcome == DiscoveryOutcome.Resolved ? chosen : Unbound,
                    matched));
            }

            return results;
        }

        /// <summary>
        /// Looks for a series whose <b>ending</b> identifies the channel, whatever the application is called.
        /// </summary>
        /// <returns>How many evidenced candidates were found this way.</returns>
        /// <remarks>
        /// Returns the pick rather than taking <c>ref MetricCandidate chosen</c>: a <c>ref</c> parameter is
        /// illegal on an async method, and this became async when the evidence callback did. Same contract,
        /// one fewer way to forget to assign it.
        /// </remarks>
        private static async Task<(int Found, MetricCandidate Chosen)> BySuffixAsync(
            MetricIndex metric,
            IReadOnlySet<string> available,
            Func<string, Task<int>> podsReporting,
            List<MetricCandidate> matched,
            MetricCandidate chosen)
        {
            var suffixes = MetricNameCatalog.SuffixesFor(metric);

            if (suffixes.Count == 0)
            {
                return (0, chosen);
            }

            var quantile = metric switch
            {
                MetricIndex.LatencyP50Ms => 0.50,
                MetricIndex.LatencyP95Ms => 0.95,
                MetricIndex.LatencyP99Ms => 0.99,
                _ => double.NaN,
            };

            var found = 0;

            foreach (var name in available)
            {
                if (IsInfrastructure(name) || !EndsWithAny(name, suffixes) || AlreadyConsidered(matched, name))
                {
                    continue;
                }

                var pods = await podsReporting(name).ConfigureAwait(false);

                if (pods <= 0)
                {
                    // Recorded anyway: "this series exists and your pods do not export it" is a different
                    // problem from "nothing like this exists", and only the report can tell them apart.
                    matched.Add(new MetricCandidate(
                        name, MetricNameCatalog.KindOf(name), string.Empty, 0, quantile, Inferred: true));

                    continue;
                }

                var kind = MetricNameCatalog.KindOf(name);

                // A histogram is bound by its family name; the _bucket series is only how it was found.
                var source = kind == MetricSourceKind.HistogramSeconds
                    ? name[..^"_bucket".Length]
                    : name;

                var candidate = new MetricCandidate(source, kind, string.Empty, pods, quantile, Inferred: true);

                matched.Add(candidate);
                found++;

                if (found == 1)
                {
                    chosen = candidate;
                }
            }

            return (found, chosen);
        }

        /// <summary>
        /// Whether a series belongs to the cluster's own exporters rather than to an application.
        ///
        /// <para><b>Found by running this against a real cluster, not by reasoning.</b> The
        /// <c>_failures_total</c> suffix, meant to catch an application's error counter, matched
        /// <c>container_memory_failures_total</c> — a cAdvisor page-fault counter — and offered it as a
        /// candidate for the error rate beside the real one. These namespaces are owned by cAdvisor,
        /// kube-state-metrics and node-exporter; their names do not vary per application, so they are already
        /// matched exactly where they matter and can only add noise to a suffix rule.</para>
        /// </summary>
        private static bool IsInfrastructure(string name)
        {
            return name.StartsWith("container_", StringComparison.Ordinal)
                   || name.StartsWith("kube_", StringComparison.Ordinal)
                   || name.StartsWith("kubelet_", StringComparison.Ordinal)
                   || name.StartsWith("node_", StringComparison.Ordinal)
                   || name.StartsWith("apiserver_", StringComparison.Ordinal)
                   || name.StartsWith("etcd_", StringComparison.Ordinal);
        }

        private static bool EndsWithAny(string name, IReadOnlyList<string> suffixes)
        {
            for (var i = 0; i < suffixes.Count; i++)
            {
                if (name.EndsWith(suffixes[i], StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool AlreadyConsidered(List<MetricCandidate> matched, string name)
        {
            for (var i = 0; i < matched.Count; i++)
            {
                if (string.Equals(matched[i].Source, name, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Names the runtime families present among <paramref name="available"/>, most specific evidence
        /// first — what an operator would call "what these pods are".
        /// </summary>
        public static IReadOnlyList<string> Stacks(IReadOnlySet<string> available)
        {
            ArgumentNullException.ThrowIfNull(available);

            var seen = new List<string>();

            foreach (var name in available)
            {
                var stack = MetricNameCatalog.StackOf(name);

                if (stack.Length == 0 || Contains(seen, stack))
                {
                    continue;
                }

                seen.Add(stack);
            }

            seen.Sort(StringComparer.Ordinal);

            return seen;
        }

        /// <summary>
        /// Renders the proposal as the <c>metrics</c> block of a guard configuration file.
        ///
        /// <para>Only the resolved channels are written. An ambiguous one is left out on purpose so that the
        /// file cannot silently encode a guess — the report says what was ambiguous and why, and the operator
        /// adds it deliberately or not at all.</para>
        /// </summary>
        public static string ToConfigJson(IReadOnlyList<ChannelDiscovery> discovered)
        {
            ArgumentNullException.ThrowIfNull(discovered);

            var text = new StringBuilder();

            text.Append("  \"metrics\": {\n");

            var written = 0;

            for (var i = 0; i < discovered.Count; i++)
            {
                if (discovered[i].Outcome != DiscoveryOutcome.Resolved)
                {
                    continue;
                }

                var chosen = discovered[i].Chosen;

                if (written > 0)
                {
                    text.Append(",\n");
                }

                text.Append("    \"").Append(discovered[i].Metric).Append("\": { \"source\": \"")
                    .Append(chosen.Source).Append("\", \"kind\": \"").Append(chosen.Kind).Append('"');

                if (double.IsFinite(chosen.Quantile))
                {
                    text.Append(", \"quantile\": ").Append(chosen.Quantile.ToString("0.00",
                        System.Globalization.CultureInfo.InvariantCulture));
                }

                text.Append(" }");
                written++;
            }

            text.Append("\n  }");

            return text.ToString();
        }

        private static bool Contains(List<string> values, string value)
        {
            for (var i = 0; i < values.Count; i++)
            {
                if (string.Equals(values[i], value, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
