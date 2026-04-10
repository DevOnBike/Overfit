// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring.Contracts
{
    /// <summary>Per-feature statistics computed by <see cref="TrainingDataAnalyzer"/>.</summary>
    public sealed record FeatureReport
    {
        /// <summary>Zero-based index into the feature vector.</summary>
        public int Index { get; init; }

        /// <summary>Human-readable name (e.g. "CpuUsageRatio.mean").</summary>
        public string Name { get; init; } = string.Empty;

        public float Mean { get; init; }
        public float Std { get; init; }
        public float Min { get; init; }
        public float Max { get; init; }

        /// <summary>
        /// Std / |Mean|. Low values (&lt; 0.01) indicate the feature barely changes
        /// across the training set — the model cannot learn from it.
        /// </summary>
        public float CoefficientOfVariation { get; init; }

        /// <summary>True when CV &lt; <see cref="TrainingDataAnalyzerConfig.ConstantFeatureThreshold"/>.</summary>
        public bool IsConstant { get; init; }

        /// <summary>Number of NaN or Inf values in this feature dimension.</summary>
        public int NonFiniteCount { get; init; }
    }

}