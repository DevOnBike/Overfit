// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>Immutable configuration for <see cref="TrainingDataAnalyzer" />.</summary>
    public sealed record TrainingDataAnalyzerConfig
    {
        /// <summary>
        ///     Minimum number of training vectors required for viable training.
        ///     Below this count the model cannot generalise. Default: 200.
        /// </summary>
        public int MinSamples { get; init; } = 200;

        /// <summary>
        ///     Coefficient of variation below which a feature is considered constant.
        ///     CV = std / |mean|. Default: 0.01 (1 %).
        /// </summary>
        public float ConstantFeatureThreshold { get; init; } = 0.01f;

        /// <summary>
        ///     Maximum fraction of features allowed to be constant before the dataset
        ///     is considered non-viable. Default: 0.5 (50 %).
        /// </summary>
        public float MaxConstantFeatureFraction { get; init; } = 0.5f;

        /// <summary>
        ///     Pearson |r| above which two features are reported as highly correlated.
        ///     Default: 0.90.
        /// </summary>
        public float HighCorrelationThreshold { get; init; } = 0.90f;

        /// <summary>
        ///     When true, the correlation matrix is computed (O(F² × N)).
        ///     Disable for very large datasets where speed matters. Default: true.
        /// </summary>
        public bool ComputeCorrelation { get; init; } = true;
    }
}