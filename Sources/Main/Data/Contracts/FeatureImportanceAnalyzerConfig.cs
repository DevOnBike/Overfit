// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>Immutable configuration for <see cref="FeatureImportanceAnalyzer" />.</summary>
    public sealed record FeatureImportanceAnalyzerConfig
    {
        /// <summary>
        ///     Number of permutation iterations per feature.
        ///     More iterations → more stable estimates, slower runtime.
        ///     Default: 20 (good balance for 48-dim vectors, ~2875 training samples).
        /// </summary>
        public int Iterations { get; init; } = 20;

        /// <summary>
        ///     Number of training samples used per iteration.
        ///     Subsetting speeds up analysis when the training set is large.
        ///     null = use all samples. Default: 500.
        /// </summary>
        public int? SamplesPerIteration { get; init; } = 500;

        /// <summary>
        ///     Z-score threshold above which a feature is classified as Confirmed.
        ///     Default: 2.0 — feature must beat shadow by ≥ 2σ.
        /// </summary>
        public float ConfirmThreshold { get; init; } = 2.0f;

        /// <summary>
        ///     Z-score threshold below which a feature is classified as Rejected.
        ///     Default: -2.0 — feature consistently worse than shadow.
        /// </summary>
        public float RejectThreshold { get; init; } = -2.0f;

        /// <summary>Random seed for permutation reproducibility. null = non-deterministic.</summary>
        public int? Seed { get; init; }
    }
}