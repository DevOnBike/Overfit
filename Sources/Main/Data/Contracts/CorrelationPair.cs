// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>
    /// Pair of features whose Pearson correlation exceeds
    /// <see cref="TrainingDataAnalyzerConfig.HighCorrelationThreshold"/>.
    /// </summary>
    public sealed record CorrelationPair
    {
        public int FeatureIndexA { get; init; }
        public int FeatureIndexB { get; init; }
        public string FeatureNameA { get; init; } = string.Empty;
        public string FeatureNameB { get; init; } = string.Empty;

        /// <summary>Pearson r ∈ [-1, 1].</summary>
        public float Correlation { get; init; }
    }
}