// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;

namespace DevOnBike.Overfit.Anomalies.Gpt
{
    /// <summary>
    /// Converts MetricSnapshot sequences to integer token sequences for GPT training.
    ///
    /// Vocabulary: 12 metrics × 64 bins = 768 tokens.
    /// Token for metric m, bin b = m * BinsPerMetric + b.
    ///
    /// One snapshot → 12 tokens (TokensPerSnapshot).
    /// Context of 252 tokens = 21 snapshots = ~5 minutes at 15s scrape interval.
    ///
    /// Binning:
    ///   Ratio metrics [0,1]: uniform bins
    ///   Byte/latency/request metrics: log-scale (handles wide dynamic range)
    /// </summary>
    public sealed class MetricTokenizer
    {
        public const int MetricCount       = MetricSnapshot.FeatureCount; // 12
        public const int BinsPerMetric     = 64;
        public const int VocabSize         = MetricCount * BinsPerMetric; // 768
        public const int TokensPerSnapshot = MetricCount;                  // 12

        // Metric index constants — must match MetricSnapshot.WriteFeatureVector order
        private const int IdxCpuUsage       = 0;
        private const int IdxCpuThrottle    = 1;
        private const int IdxMemoryBytes    = 2;
        private const int IdxOomRate        = 3;
        private const int IdxLatencyP50     = 4;
        private const int IdxLatencyP95     = 5;
        private const int IdxLatencyP99     = 6;
        private const int IdxRequestsPerSec = 7;
        private const int IdxErrorRate      = 8;
        private const int IdxGcHeapBytes    = 9;
        private const int IdxGcPauseRatio   = 10;
        private const int IdxThreadPoolQueue= 11;

        // Per-metric (maxValue, isLogScale)
        private static readonly (float Max, bool LogScale)[] MetricRanges =
        [
            (1.0f,    false), // CpuUsageRatio
            (1.0f,    false), // CpuThrottleRatio
            (8e9f,    true),  // MemoryWorkingSetBytes
            (0.1f,    true),  // OomEventsRate
            (2000f,   true),  // LatencyP50Ms
            (5000f,   true),  // LatencyP95Ms
            (10000f,  true),  // LatencyP99Ms
            (1000f,   true),  // RequestsPerSecond
            (1.0f,    false), // ErrorRate
            (2e10f,   true),  // GcGen2HeapBytes
            (1.0f,    false), // GcPauseRatio
            (500f,    false), // ThreadPoolQueueLength
        ];

        /// <summary>
        /// Encodes a single MetricSnapshot into TokensPerSnapshot (12) tokens.
        /// MetricSnapshot is a readonly struct — pass by value to avoid CS8156.
        /// </summary>
        public void EncodeSnapshot(MetricSnapshot snapshot, int[] destination, int offset = 0)
        {
            if (destination.Length - offset < TokensPerSnapshot)
            {
                throw new ArgumentException(
                    $"Destination too short: need {TokensPerSnapshot}, have {destination.Length - offset}.");
            }

            Span<float> features = stackalloc float[MetricCount];
            snapshot.WriteFeatureVector(features);

            for (var m = 0; m < MetricCount; m++)
            {
                destination[offset + m] = m * BinsPerMetric + Quantize(features[m], m);
            }
        }

        /// <summary>
        /// Encodes a sequence of snapshots into a flat token array.
        /// Length = snapshots.Count × TokensPerSnapshot.
        /// </summary>
        public int[] EncodeSequence(IReadOnlyList<MetricSnapshot> snapshots)
        {
            var tokens = new int[snapshots.Count * TokensPerSnapshot];
            for (var i = 0; i < snapshots.Count; i++)
            {
                EncodeSnapshot(snapshots[i], tokens, i * TokensPerSnapshot);
            }
            return tokens;
        }

        /// <summary>Metric index (0-11) for a token.</summary>
        public static int MetricIndexOf(int token) => token / BinsPerMetric;

        /// <summary>Bin index (0-63) for a token.</summary>
        public static int BinOf(int token) => token % BinsPerMetric;

        /// <summary>
        /// Approximate metric value at bin midpoint.
        /// Used for anomaly explanation: "expected CPU ~42%, got 87%".
        /// </summary>
        public static float Decode(int token)
        {
            var m   = MetricIndexOf(token);
            var bin = BinOf(token);
            var (max, logScale) = MetricRanges[m];
            var ratio = (bin + 0.5f) / BinsPerMetric;
            return logScale
                ? MathF.Exp(ratio * MathF.Log(1f + max)) - 1f
                : ratio * max;
        }

        /// <summary>Human-readable metric name for a token.</summary>
        public static string MetricNameOf(int token) =>
            MetricIndexOf(token) switch
            {
                IdxCpuUsage        => "cpu_usage_ratio",
                IdxCpuThrottle     => "cpu_throttle_ratio",
                IdxMemoryBytes     => "memory_working_set_bytes",
                IdxOomRate         => "oom_events_rate",
                IdxLatencyP50      => "latency_p50_ms",
                IdxLatencyP95      => "latency_p95_ms",
                IdxLatencyP99      => "latency_p99_ms",
                IdxRequestsPerSec  => "requests_per_second",
                IdxErrorRate       => "error_rate",
                IdxGcHeapBytes     => "gc_gen2_heap_bytes",
                IdxGcPauseRatio    => "gc_pause_ratio",
                IdxThreadPoolQueue => "threadpool_queue_length",
                _ => $"metric_{MetricIndexOf(token)}"
            };

        // ── Private ──────────────────────────────────────────────────────────

        private static int Quantize(float value, int m)
        {
            var (max, logScale) = MetricRanges[m];
            if (value <= 0f) return 0;
            if (value >= max) return BinsPerMetric - 1;
            var ratio = logScale
                ? MathF.Log(1f + value) / MathF.Log(1f + max)
                : value / max;
            return Math.Clamp((int)(ratio * BinsPerMetric), 0, BinsPerMetric - 1);
        }
    }
}
