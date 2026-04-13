// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    public sealed class MonitoringPipelineOptions
    {
        /// <summary>Must match AlignerOptions.WindowSize.</summary>
        public int WindowSize { get; init; } = 60;

        /// <summary>Must match AlignerOptions.StepSeconds.</summary>
        public int StepSeconds { get; init; } = 15;

        /// <summary>Must match AlignerOptions.MetricCount.</summary>
        public int MetricCount { get; init; } = 12;

        /// <summary>Must match AlignerOptions.MaxGapSteps.</summary>
        public int MaxGapSteps { get; init; } = 2;

        /// <summary>Must match SanitizerOptions.WarmupDuration.</summary>
        public TimeSpan WarmupDuration { get; init; } = TimeSpan.FromMinutes(5);

        /// <summary>Must match SanitizerOptions.MaxNanRatio.</summary>
        public float MaxNanRatio { get; init; } = 0.1f;
    }
}