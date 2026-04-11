// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>
    ///     Full analysis report returned by <see cref="TrainingDataAnalyzer.Analyze" />.
    /// </summary>
    public sealed record TrainingDataReport
    {
        /// <summary>Number of training vectors analysed.</summary>
        public int SampleCount { get; init; }

        /// <summary>Dimensionality of each vector.</summary>
        public int FeatureDimension { get; init; }

        /// <summary>Per-dimension statistics. Length == FeatureDimension.</summary>
        public IReadOnlyList<FeatureReport> FeatureReports { get; init; } = [];

        /// <summary>
        ///     Feature pairs whose |r| exceeds the configured threshold.
        ///     High correlation indicates redundancy — consider removing one of the pair.
        /// </summary>
        public IReadOnlyList<CorrelationPair> HighCorrelationPairs { get; init; } = [];

        /// <summary>
        ///     Human-readable warnings. Empty list means data looks clean.
        ///     Warnings do not block training — they inform the operator.
        /// </summary>
        public IReadOnlyList<string> Warnings { get; init; } = [];

        /// <summary>
        ///     Hard errors that make training pointless or dangerous.
        ///     When non-empty, <see cref="IsViableForTraining" /> is false.
        /// </summary>
        public IReadOnlyList<string> Errors { get; init; } = [];

        /// <summary>
        ///     True when there are no hard errors and the data is considered
        ///     suitable for training. Warnings may still be present.
        /// </summary>
        public bool IsViableForTraining => Errors.Count == 0;

        /// <summary>Number of features flagged as effectively constant.</summary>
        public int ConstantFeatureCount =>
            FeatureReports.Count(f => f.IsConstant);

        /// <summary>Total non-finite values across all features and samples.</summary>
        public int TotalNonFiniteCount =>
            FeatureReports.Sum(f => f.NonFiniteCount);
    }
}