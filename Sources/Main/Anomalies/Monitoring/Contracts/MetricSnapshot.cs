// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    /// <summary>
    /// A single metric sample for one pod.
    /// Readonly struct — zero allocations when passing through the pipeline.
    /// Feature layout (must match <see cref="WriteFeatureVector"/> order):
    ///   [0]  CpuUsageRatio          — rate / cpu_limit
    ///   [1]  CpuThrottleRatio       — throttled_periods / total_periods
    ///   [2]  MemoryWorkingSetBytes  — container_memory_working_set_bytes
    ///   [3]  OomEventsRate          — rate(container_oom_events_total, 1m)
    ///   [4]  LatencyP50Ms           — histogram_quantile(0.50) * 1000
    ///   [5]  LatencyP95Ms           — histogram_quantile(0.95) * 1000
    ///   [6]  LatencyP99Ms           — histogram_quantile(0.99) * 1000
    ///   [7]  RequestsPerSecond      — rate(http_server_request_duration_seconds_count, 1m)
    ///   [8]  ErrorRate              — rate(5xx) / rate(total)
    ///   [9]  GcGen2HeapBytes        — gc_heap_size_bytes{generation="2"}
    ///   [10] GcPauseRatio           — rate(gc_pause_total_seconds, 1m)
    ///   [11] ThreadPoolQueueLength  — dotnet_threadpool_queue_length
    ///
    /// Changing FeatureCount invalidates any trained model — full retrain required.
    /// </summary>
    public readonly struct MetricSnapshot
    {
        public DateTime Timestamp { get; init; }
        public string PodName { get; init; }

        // --- compute ---

        /// <summary>
        /// CPU usage as a fraction of the pod's CPU limit. [0, 1]
        /// Normalised to the limit — 0.4 on a 0.5-core pod is near saturation,
        /// 0.4 on a 4-core pod is relaxed.
        /// </summary>
        public float CpuUsageRatio { get; init; }

        /// <summary>
        /// Fraction of CFS scheduler periods during which the container was CPU-throttled. [0, 1]
        /// A pod can be throttled even at 60 % usage when its CPU limit is too low.
        /// Rises 1–2 minutes before latency starts to degrade — early warning signal.
        /// </summary>
        public float CpuThrottleRatio { get; init; }

        /// <summary>
        /// Working set RSS in bytes — excludes reclaimable page cache.
        /// K8s uses this value for pod eviction. Do not use container_memory_usage_bytes
        /// which includes page cache and is always misleadingly higher.
        /// </summary>
        public float MemoryWorkingSetBytes { get; init; }

        /// <summary>
        /// OOM kill event rate (events/s). Normally always 0.
        /// Any non-zero value is a hard signal — also consider an unconditional alert
        /// independent of the anomaly score.
        /// </summary>
        public float OomEventsRate { get; init; }

        // --- HTTP/gRPC traffic ---

        /// <summary>
        /// p50 request latency in milliseconds.
        /// The p99/p50 ratio is a sensitive indicator — normally 3–5×;
        /// rising above 15× signals lock contention or downstream degradation.
        /// </summary>
        public float LatencyP50Ms { get; init; }

        /// <summary>
        /// p95 request latency in milliseconds. Primary SLA indicator.
        /// The autoencoder learns the normal correlation of p95 with RPS —
        /// p95 rising without a corresponding RPS increase is a strong anomaly signal.
        /// </summary>
        public float LatencyP95Ms { get; init; }

        /// <summary>
        /// p99 request latency in milliseconds.
        /// Most sensitive to GC pauses, lock contention and downstream timeouts.
        /// Never use average latency (sum/count) — it masks the long tail.
        /// </summary>
        public float LatencyP99Ms { get; init; }

        /// <summary>Requests per second. Context axis for all other metrics.</summary>
        public float RequestsPerSecond { get; init; }

        /// <summary>
        /// Fraction of 5xx responses. [0, 1]
        /// 0–0.5 % is normal background noise. Sustained above 1 % is an anomaly.
        /// </summary>
        public float ErrorRate { get; init; }

        // --- .NET runtime ---

        /// <summary>
        /// Gen2 managed heap size in bytes.
        /// Best early signal for memory leaks — Gen2 objects are long-lived.
        /// Monotonic growth over 30+ minutes with no GC collection indicates a leak.
        /// </summary>
        public float GcGen2HeapBytes { get; init; }

        /// <summary>
        /// Fraction of wall-clock time spent in GC pauses. [0, 1]
        /// Normalised — stays stable under varying load unlike raw gc_pause_ms.
        /// Rising ratio at constant RPS = GC thrashing, classic symptom of memory pressure.
        /// </summary>
        public float GcPauseRatio { get; init; }

        /// <summary>
        /// Number of pending work items in the ThreadPool queue.
        /// Normally below 20. Early signal for thread starvation and deadlocks —
        /// the queue grows before latency starts to rise dramatically.
        /// </summary>
        public float ThreadPoolQueueLength { get; init; }

        /// <summary>
        /// Feature vector size — pipeline and model contract.
        /// InputSize for the autoencoder = FeatureCount × FeatureExtractor.StatsPerFeature = 12 × 4 = 48.
        /// Changing this requires rebuilding the autoencoder weights.
        /// </summary>
        public const int FeatureCount = 12;

        /// <summary>
        /// Writes the feature vector to the specified <paramref name="destination"/>.
        /// Zero allocation — the only allowed path in the production pipeline.
        /// The order is fixed and must match the encoder's expectations.
        /// </summary>
        public void WriteFeatureVector(Span<float> destination)
        {
            if (destination.Length < FeatureCount)
            {
                throw new ArgumentException($"Destination too short: need {FeatureCount}, got {destination.Length}.", nameof(destination));
            }

            destination[0] = CpuUsageRatio;
            destination[1] = CpuThrottleRatio;
            destination[2] = MemoryWorkingSetBytes;
            destination[3] = OomEventsRate;
            destination[4] = LatencyP50Ms;
            destination[5] = LatencyP95Ms;
            destination[6] = LatencyP99Ms;
            destination[7] = RequestsPerSecond;
            destination[8] = ErrorRate;
            destination[9] = GcGen2HeapBytes;
            destination[10] = GcPauseRatio;
            destination[11] = ThreadPoolQueueLength;
        }
    }
}