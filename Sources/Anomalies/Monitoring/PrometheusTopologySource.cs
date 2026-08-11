// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring.Abstractions;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Reads pod ownership and placement from kube-state-metrics, through Prometheus.
    ///
    /// <para><b>The relationships are in the LABELS, not the values.</b> kube-state-metrics emits a constant
    /// <c>1</c> and carries the fact in <c>owner_kind</c>, <c>owner_name</c>, <c>node</c>. Every other reader
    /// in this project parses for a value and would come back with a column of ones, which is why none of
    /// them could be reused and why this class exists at all.</para>
    ///
    /// <para><b>Two hops, because Kubernetes has two.</b> A Deployment's pod is owned by a ReplicaSet, and the
    /// ReplicaSet by the Deployment. Resolving only the first hop names the <i>version</i> — which is useful,
    /// and is not what a human calls the workload. Both are kept: during a rollout two ReplicaSets serve at
    /// once, and comparing across them compares two different builds, which is why the grouper scores
    /// <c>SameReplicaSet</c> above <c>SameWorkload</c>.</para>
    ///
    /// <para><b>A pod owned directly by something else keeps that owner as its workload.</b> A StatefulSet
    /// pod, a Job pod or a bare pod has no ReplicaSet, and inventing one would be worse than reporting what
    /// is actually there.</para>
    ///
    /// <para><b>A failed refresh leaves the previous snapshot in place</b> rather than emptying it. An empty
    /// topology is not neutral — it makes every pod share an empty workload and therefore merge with every
    /// other, which is the failure this class was built to prevent. Stale coordinates are wrong slowly;
    /// blank ones are wrong instantly and in the worst direction.</para>
    /// </summary>
    public sealed class PrometheusTopologySource : IRefreshablePodTopology, IPodRoster, IDisposable
    {
        private static readonly JsonSerializerOptions _jsonOptions = new()
        {
            PropertyNameCaseInsensitive = true,
        };

        private readonly IClock _clock;
        private readonly IPrometheusQuerySelector _selector;
        private readonly string _peerGroupLabel;
        private readonly string _baseUrl;
        private readonly HttpClient _http;
        private readonly bool _ownsHttpClient;

        private Dictionary<string, PodPlacement> _snapshot = new(StringComparer.Ordinal);
        private bool _disposed;

        /// <param name="prometheusBaseUrl">HTTP API base URL. The only thing this type needs to reach the
        /// cluster: topology comes from kube-state-metrics through Prometheus, not from the API server.</param>
        /// <param name="selector">Supplies the namespace and pod filter, so this type and the metric source
        /// cannot drift into watching different sets of pods.</param>
        /// <param name="httpClient">Optional shared client. One is created and owned when omitted; a client
        /// that is lent is not disposed here.</param>
        /// <param name="peerGroupLabel">
        /// Pod label naming which replicas may be compared against each other — <c>role</c> for most database
        /// and queue operators. Empty means none is declared and every pod compares against every other,
        /// which is the behaviour before this existed.
        ///
        /// <para>Declared rather than inferred because a rollout, a canary and an elected leader produce the
        /// same shape and want opposite answers. Read from the cluster rather than from a list because the
        /// operator already publishes it and updates it on failover; a hand-written list is wrong from the
        /// first election.</para>
        /// </param>
        /// <param name="clock">
        /// Time source. Defaults to <see cref="SystemClock"/>.
        ///
        /// <para>Read in one place — stamping <see cref="LastRefreshed"/> after a refresh that resolved pods.
        /// Nothing about resolution or grouping depends on it, so injecting one changes what this type
        /// REPORTS about its own freshness and never what it answers.</para>
        /// </param>
        public PrometheusTopologySource(
            string prometheusBaseUrl,
            IPrometheusQuerySelector selector,
            HttpClient? httpClient = null,
            string peerGroupLabel = "",
            IClock? clock = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(prometheusBaseUrl);
            ArgumentNullException.ThrowIfNull(selector);

            _clock = clock ?? SystemClock.Instance;
            _baseUrl = prometheusBaseUrl.TrimEnd('/');
            _selector = selector;
            _peerGroupLabel = peerGroupLabel ?? string.Empty;
            _ownsHttpClient = httpClient == null;
            _http = httpClient ?? new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
        }

        /// <summary>Pods in the current snapshot. Zero means nothing has been resolved yet.</summary>
        public int Count => _snapshot.Count;

        /// <summary>
        /// Reads the declared peer group for every pod from <c>kube_pod_labels</c>.
        ///
        /// <para>A pod that does not carry the label gets no entry, and therefore an empty group. That is
        /// deliberate rather than a gap: an empty group is shared with every other unlabelled pod, so they
        /// keep comparing against each other exactly as they did before anyone declared anything. The
        /// alternative — inventing a group per pod — would leave each one alone and silently switch the peer
        /// family off for the whole deployment.</para>
        /// </summary>
        private async Task<Dictionary<string, string>> ReadPeerGroupsAsync(CancellationToken ct)
        {
            var series = PromqlCatalog.PodLabelSeriesName(_peerGroupLabel);
            var rows = await QueryAsync(PromqlCatalog.PodLabelsQuery(_selector), ct).ConfigureAwait(false);
            var groups = new Dictionary<string, string>(rows.Count, StringComparer.Ordinal);

            for (var i = 0; i < rows.Count; i++)
            {
                var labels = rows[i];

                if (Label(labels, "pod") is { Length: > 0 } pod
                    && Label(labels, series) is { Length: > 0 } group)
                {
                    groups[pod] = group;
                }
            }

            return groups;
        }

        /// <inheritdoc/>
        public bool TryResolve(string pod, out PodPlacement placement)
        {
            ArgumentNullException.ThrowIfNull(pod);

            return _snapshot.TryGetValue(pod, out placement);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Straight off the last snapshot, which comes from <c>kube_pod_owner</c> and friends — the cluster's
        /// own list, not a list of whoever happened to export a metric. That distinction is the entire value
        /// of this member: the two lists differing is the only evidence a pod exists and is saying nothing.
        /// </remarks>
        public IReadOnlyList<string> KnownPods
        {
            get
            {
                var snapshot = _snapshot;
                var pods = new string[snapshot.Count];
                var i = 0;

                foreach (var pod in snapshot.Keys)
                {
                    pods[i++] = pod;
                }

                return pods;
            }
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Written by <see cref="RefreshAsync"/> only where it also replaces the snapshot, which is the point:
        /// a failed refresh keeps the old pod list on purpose, and if this advanced alongside it the caller
        /// would be told an hour-old list is current.
        /// </remarks>
        public DateTimeOffset? LastRefreshed
        {
            get;
            private set;
        }

        /// <summary>
        /// Re-reads ownership and placement. Returns how many pods were resolved; on any failure the previous
        /// snapshot is kept and <c>-1</c> is returned, so a caller can report degraded topology without
        /// having its grouping silently collapse.
        /// </summary>
        public async Task<int> RefreshAsync(CancellationToken ct)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            try
            {
                var podOwners = await QueryAsync(PromqlCatalog.PodOwnershipQuery(_selector), ct)
                    .ConfigureAwait(false);
                var replicaSetOwners = await QueryAsync(PromqlCatalog.ReplicaSetOwnershipQuery(_selector), ct)
                    .ConfigureAwait(false);
                var podNodes = await QueryAsync(PromqlCatalog.PodNodeQuery(_selector), ct)
                    .ConfigureAwait(false);
                var podCreated = await QueryValuesAsync(PromqlCatalog.PodCreatedQuery(_selector), ct)
                    .ConfigureAwait(false);

                // Only issued when a label was named. Asking for kube_pod_labels on every refresh would cost
                // a query per cycle to fill a field nobody configured, and the series is one of the widest
                // kube-state-metrics produces.
                var peerGroupOf = _peerGroupLabel.Length > 0
                    ? await ReadPeerGroupsAsync(ct).ConfigureAwait(false)
                    : null;

                // ReplicaSet -> Deployment, the second hop.
                var deploymentOf = new Dictionary<string, string>(StringComparer.Ordinal);

                for (var i = 0; i < replicaSetOwners.Count; i++)
                {
                    var labels = replicaSetOwners[i];

                    if (Label(labels, "replicaset") is { Length: > 0 } replicaSet
                        && Label(labels, "owner_name") is { Length: > 0 } owner)
                    {
                        deploymentOf[replicaSet] = owner;
                    }
                }

                // Pod -> creation time. Absent for a pod kube-state-metrics has not caught up with, which
                // reads as "age unknown" downstream rather than as "brand new" — guessing young would
                // silence the trend family on a pod that may have been running for a week.
                var createdOf = new Dictionary<string, DateTimeOffset>(StringComparer.Ordinal);

                for (var i = 0; i < podCreated.Count; i++)
                {
                    var (createdLabels, createdValue) = podCreated[i];

                    if (Label(createdLabels, "pod") is { Length: > 0 } createdPod && createdValue > 0.0)
                    {
                        createdOf[createdPod] = DateTimeOffset.FromUnixTimeSeconds((long)createdValue);
                    }
                }

                var nodeOf = new Dictionary<string, string>(StringComparer.Ordinal);

                for (var i = 0; i < podNodes.Count; i++)
                {
                    var labels = podNodes[i];

                    if (Label(labels, "pod") is { Length: > 0 } pod
                        && Label(labels, "node") is { Length: > 0 } node)
                    {
                        nodeOf[pod] = node;
                    }
                }

                var resolved = new Dictionary<string, PodPlacement>(
                    podOwners.Count, StringComparer.Ordinal);

                for (var i = 0; i < podOwners.Count; i++)
                {
                    var labels = podOwners[i];
                    var pod = Label(labels, "pod");

                    if (pod.Length == 0)
                    {
                        continue;
                    }

                    var owner = Label(labels, "owner_name");
                    var kind = Label(labels, "owner_kind");

                    // A ReplicaSet owner resolves one more hop to the Deployment; anything else — StatefulSet,
                    // Job, DaemonSet — already names the workload, so it is kept as it stands.
                    var workload = string.Equals(kind, "ReplicaSet", StringComparison.Ordinal)
                                   && deploymentOf.TryGetValue(owner, out var deployment)
                        ? deployment
                        : owner;

                    var replicaSet = string.Equals(kind, "ReplicaSet", StringComparison.Ordinal)
                        ? owner
                        : string.Empty;

                    var peerGroup = peerGroupOf != null
                                    && peerGroupOf.TryGetValue(pod, out var declared)
                        ? declared
                        : string.Empty;

                    resolved[pod] = new PodPlacement(
                        workload,
                        replicaSet,
                        nodeOf.TryGetValue(pod, out var node) ? node : string.Empty,
                        peerGroup,
                        createdOf.TryGetValue(pod, out var created) ? created : default);
                }

                if (resolved.Count == 0)
                {
                    // Prometheus answered and matched nothing. Almost always kube-state-metrics is absent or
                    // the namespace matcher is wrong — and adopting an empty topology would merge the whole
                    // namespace into one incident, so the previous snapshot stands.
                    return -1;
                }

                _snapshot = resolved;

                // Stamped here and nowhere else — every other exit from this method leaves the previous
                // snapshot in place, and a timestamp that moved with them would describe an attempt rather
                // than a list.
                LastRefreshed = _clock.UtcNow;

                return resolved.Count;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception)
            {
                return -1;
            }
        }

        /// <summary>
        /// Runs an instant query and returns each series' <b>label set</b>. The value is deliberately
        /// discarded — for these metrics it is always 1 and carries nothing.
        /// </summary>
        private async Task<List<Dictionary<string, string>>> QueryAsync(string promql, CancellationToken ct)
        {
            var url = $"{_baseUrl}/api/v1/query?query={Uri.EscapeDataString(promql)}";

            using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            var result = new List<Dictionary<string, string>>();

            using var document = JsonDocument.Parse(body);

            if (!document.RootElement.TryGetProperty("data", out var data)
                || !data.TryGetProperty("result", out var series))
            {
                return result;
            }

            foreach (var entry in series.EnumerateArray())
            {
                if (!entry.TryGetProperty("metric", out var metric))
                {
                    continue;
                }

                var labels = new Dictionary<string, string>(StringComparer.Ordinal);

                foreach (var label in metric.EnumerateObject())
                {
                    labels[label.Name] = label.Value.GetString() ?? string.Empty;
                }

                result.Add(labels);
            }

            return result;
        }

        /// <summary>
        /// The same instant query, keeping the sample <b>value</b> alongside the labels.
        ///
        /// <para>Separate from <see cref="QueryAsync"/> rather than replacing it: for the ownership and
        /// placement metrics the value is always 1 and discarding it is the honest thing to do. For
        /// <c>kube_pod_created</c> the value <i>is</i> the answer.</para>
        /// </summary>
        private async Task<List<(Dictionary<string, string> Labels, double Value)>> QueryValuesAsync(
            string promql, CancellationToken ct)
        {
            var url = $"{_baseUrl}/api/v1/query?query={Uri.EscapeDataString(promql)}";

            using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            var result = new List<(Dictionary<string, string>, double)>();

            using var document = JsonDocument.Parse(body);

            if (!document.RootElement.TryGetProperty("data", out var data)
                || !data.TryGetProperty("result", out var series))
            {
                return result;
            }

            foreach (var entry in series.EnumerateArray())
            {
                if (!entry.TryGetProperty("metric", out var metric)
                    || !entry.TryGetProperty("value", out var sample)
                    || sample.GetArrayLength() < 2)
                {
                    continue;
                }

                var labels = new Dictionary<string, string>(StringComparer.Ordinal);

                foreach (var label in metric.EnumerateObject())
                {
                    labels[label.Name] = label.Value.GetString() ?? string.Empty;
                }

                // Prometheus renders the value as a STRING in the JSON, always.
                if (double.TryParse(
                        sample[1].GetString(),
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture,
                        out var value))
                {
                    result.Add((labels, value));
                }
            }

            return result;
        }

        private static string Label(Dictionary<string, string> labels, string name)
        {
            return labels.TryGetValue(name, out var value) ? value : string.Empty;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            if (_ownsHttpClient)
            {
                _http.Dispose();
            }
        }
    }
}
