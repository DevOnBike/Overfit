// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Data.Abstractions
{
    /// <summary>
    /// Provides human-readable names for feature vector dimensions.
    ///
    /// Implement this interface to adapt <see cref="TrainingDataAnalyzer"/> and
    /// <see cref="FeatureImportanceAnalyzer"/> to any domain — not just K8s monitoring.
    ///
    /// Examples:
    ///   <see cref="MonitoringFeatureNameProvider"/> — K8s pod metrics (12 × 4 = 48 dims)
    ///   <see cref="IndexedFeatureNameProvider"/>    — generic fallback ("feature_0", "feature_1" ...)
    /// </summary>
    public interface IFeatureNameProvider
    {
        /// <summary>
        /// Returns the human-readable name for the feature at <paramref name="featureIndex"/>.
        /// Must never return null or empty string.
        /// </summary>
        string GetName(int featureIndex);

        /// <summary>
        /// Total number of dimensions this provider covers.
        /// Used for validation — the analyzer will warn if vector length != this value.
        /// Return 0 to skip dimension validation.
        /// </summary>
        int ExpectedDimension { get; }
    }
}