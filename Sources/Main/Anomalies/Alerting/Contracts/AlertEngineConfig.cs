// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com


// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com


// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com


// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Alerting.Contracts
{
    /// <summary>
    ///     Immutable configuration for <see cref="AlertEngine" />.
    /// </summary>
    public sealed record AlertEngineConfig
    {
        /// <summary>
        ///     Score at or above which an alert is dispatched.
        ///     Must be in (0, 1]. Default: 0.8.
        /// </summary>
        public float AlertThreshold { get; init; } = 0.8f;

        /// <summary>
        ///     Score at or above which severity is escalated to
        ///     <see cref="AlertSeverity.Critical" />. Must be >= AlertThreshold.
        ///     Default: 0.95.
        /// </summary>
        public float CriticalThreshold { get; init; } = 0.95f;

        /// <summary>
        ///     Minimum time between alerts for the same pod.
        ///     Prevents alert storms during sustained anomalies. Default: 5 minutes.
        /// </summary>
        public TimeSpan CooldownDuration { get; init; } = TimeSpan.FromMinutes(5);

        /// <summary>
        ///     Capacity of the internal dispatch queue.
        ///     When full, new alerts are silently dropped (DropOldest) rather than
        ///     blocking the scoring hot-path. Default: 64.
        /// </summary>
        public int DispatchQueueCapacity { get; init; } = 64;
    }

}