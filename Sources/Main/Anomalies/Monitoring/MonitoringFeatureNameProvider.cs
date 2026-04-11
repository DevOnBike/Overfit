// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Data.Abstractions;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Feature name provider for the K8s monitoring pipeline.
    /// Covers 12 raw metrics × 4 stats (mean, std, p95, delta) = 48 dimensions.
    ///
    /// Aligned with <see cref="MetricSnapshot.WriteFeatureVector"/> and
    /// <see cref="FeatureExtractor.StatsPerFeature"/>.
    /// </summary>
    public sealed class MonitoringFeatureNameProvider : IFeatureNameProvider
    {
        private static readonly string[] RawNames =
        [
            "CpuUsageRatio", "CpuThrottleRatio", "MemoryWorkingSetBytes", "OomEventsRate",
            "LatencyP50Ms",  "LatencyP95Ms",     "LatencyP99Ms",          "RequestsPerSecond",
            "ErrorRate",     "GcGen2HeapBytes",  "GcPauseRatio",          "ThreadPoolQueueLength"
        ];

        private static readonly string[] StatSuffixes = ["mean", "std", "p95", "delta"];

        private readonly int _statsPerFeature;

        /// <param name="statsPerFeature">
        ///   Stats per raw feature from <see cref="FeatureExtractor.StatsPerFeature"/>.
        ///   Default: 4 (mean, std, p95, delta).
        /// </param>
        public MonitoringFeatureNameProvider(int statsPerFeature = 4)
        {
            if (statsPerFeature <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(statsPerFeature));
            }
            _statsPerFeature = statsPerFeature;
        }

        public int ExpectedDimension => RawNames.Length * _statsPerFeature;

        public string GetName(int featureIndex)
        {
            var rawIndex = featureIndex / _statsPerFeature;
            var statIndex = featureIndex % _statsPerFeature;

            var rawName = rawIndex < RawNames.Length
                ? RawNames[rawIndex]
                : $"feature{rawIndex}";

            var statName = statIndex < StatSuffixes.Length
                ? StatSuffixes[statIndex]
                : $"stat{statIndex}";

            return $"{rawName}.{statName}";
        }
    }
}