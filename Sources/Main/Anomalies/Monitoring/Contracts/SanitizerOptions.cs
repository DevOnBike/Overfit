// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public sealed class SanitizerOptions
    {
        /// <summary>
        ///     Pods younger than this threshold are discarded.
        ///     Protects against .NET JIT warm-up false positives on pod startup.
        /// </summary>
        public TimeSpan WarmupDuration { get; init; } = TimeSpan.FromMinutes(5);

        /// <summary>
        ///     Maximum fraction of NaN values allowed in a window before the pod is discarded.
        ///     A window with more NaNs than this is considered too incomplete for reliable inference.
        /// </summary>
        public float MaxNanRatio { get; init; } = 0.1f;
    }
}