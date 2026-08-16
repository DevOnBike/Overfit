// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// Places a metric name on the cause-to-consequence axis, so callers do not have to hand-classify every
    /// signal they feed the grouper.
    ///
    /// <para>Substring matching on purpose. Prometheus metric names are compound and conventional
    /// (<c>container_cpu_cfs_throttled_periods_total</c>, <c>overfit_chat_response_time_seconds_bucket</c>),
    /// and an exact-name table would go stale the first time an exporter added a suffix. The markers below
    /// are drawn from the metric names the cluster lab actually exposes — kube-state-metrics, cAdvisor,
    /// the .NET runtime exporter and this project's own server — rather than invented.</para>
    ///
    /// <para><b>Order is load-bearing.</b> <c>container_cpu_cfs_throttled_periods_total</c> matches both
    /// <c>cpu</c> (a resource) and <c>throttl</c> (the platform acting on the workload); infrastructure is
    /// tested first because the platform's action is the actionable end of that pair.</para>
    ///
    /// <para><b>The default is <see cref="SignalClass.Symptom"/>, and that is the conservative choice.</b>
    /// Class ordering decides which finding an incident shows first, so an unrecognised metric defaulting to
    /// anything else could outrank a known restart and send the reader to the wrong place. Ranking the
    /// unknown last costs nothing when it is the only finding, because then it is shown anyway.</para>
    /// </summary>
    public static class SignalCatalog
    {
        /// <summary>The platform acting on the workload — see <see cref="SignalClass.Infrastructure"/>.</summary>
        private static readonly string[] InfrastructureMarkers =
        [
            "restart", "oom", "throttl", "evict", "terminat", "kubelet", "imagepull", "image_pull",
            "probe_fail", "node_condition", "unschedulable", "pod_status_ready", "container_status_ready",
            "crashloop", "preempt", "disruption"
        ];

        /// <summary>What the user experiences — see <see cref="SignalClass.Symptom"/>.</summary>
        private static readonly string[] SymptomMarkers =
        [
            "latency", "response_time", "duration_seconds", "ttft", "time_to_first",
            "error", "failed", "failure", "rejected", "timeout", "apdex", "http_5", "status_5",
            "throughput", "tokens_total", "requests_total"
        ];

        /// <summary>The workload's own consumption — see <see cref="SignalClass.Resource"/>.</summary>
        private static readonly string[] ResourceMarkers =
        [
            "memory", "rss", "heap", "cpu", "gc_", "_gc", "gen2", "threadpool", "thread_",
            "queue", "connection", "socket", "file_descriptor", "fd_", "disk", "working_set",
            "allocated", "pool_available", "cache", "bytes"
        ];

        /// <summary>
        /// Classifies <paramref name="signal"/>. Matching is ordinal and case-insensitive; an empty or
        /// unrecognised name yields <see cref="SignalClass.Symptom"/>.
        /// </summary>
        public static SignalClass Classify(string signal)
        {
            if (string.IsNullOrEmpty(signal))
            {
                return SignalClass.Symptom;
            }

            if (MatchesAny(signal, InfrastructureMarkers))
            {
                return SignalClass.Infrastructure;
            }

            if (MatchesAny(signal, SymptomMarkers))
            {
                return SignalClass.Symptom;
            }

            if (MatchesAny(signal, ResourceMarkers))
            {
                return SignalClass.Resource;
            }

            return SignalClass.Symptom;
        }

        private static bool MatchesAny(string signal, string[] markers)
        {
            for (var i = 0; i < markers.Length; i++)
            {
                if (signal.Contains(markers[i], StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }
    }
}