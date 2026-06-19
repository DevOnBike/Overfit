// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Gpt
{
    /// <summary>Result of scoring one MetricSnapshot.</summary>
    public sealed class AnomalyScore
    {
        /// <summary>True during warmup — window not yet filled.</summary>
        public bool IsWarmup
        {
            get; init;
        }

        /// <summary>
        /// Mean negative log-probability. ~0 = normal, ~3+ = anomaly.
        /// Tune threshold on validation data.
        /// </summary>
        public float Score
        {
            get; init;
        }

        public string PodName { get; init; } = string.Empty;
        public DateTime Timestamp
        {
            get; init;
        }
        public string WorstMetric { get; init; } = string.Empty;
        public float ExpectedValue
        {
            get; init;
        }
        public float ActualValue
        {
            get; init;
        }

        public override string ToString() =>
            IsWarmup
                ? $"[{PodName}] warmup"
                : $"[{PodName}] score={Score:F2} worst={WorstMetric} " +
                  $"expected={ExpectedValue:F1} actual={ActualValue:F1}";
    }
}
