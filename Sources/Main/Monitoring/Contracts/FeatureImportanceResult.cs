// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring.Contracts
{

    /// <summary>Importance result for one feature dimension.</summary>
    public sealed record FeatureImportanceResult
    {
        /// <summary>Zero-based index into the feature vector.</summary>
        public int Index { get; init; }

        /// <summary>Human-readable name (e.g. "GcGen2HeapBytes.mean").</summary>
        public string Name { get; init; } = string.Empty;

        /// <summary>
        /// Mean increase in MSE when this feature is randomly permuted.
        /// Higher = model relies more on this feature for reconstruction.
        /// </summary>
        public float MeanImportance { get; init; }

        /// <summary>Standard deviation of importance across permutation iterations.</summary>
        public float StdImportance { get; init; }

        /// <summary>
        /// Mean importance of the corresponding shadow (permuted) feature.
        /// Used as the baseline for the Boruta-style statistical test.
        /// </summary>
        public float ShadowMeanImportance { get; init; }

        /// <summary>
        /// How many standard deviations the feature's importance exceeds the
        /// maximum shadow importance: (MeanImportance - MaxShadowImportance) / StdImportance.
        /// Positive = consistently beats random; negative = indistinguishable from noise.
        /// </summary>
        public float ZScore { get; init; }

        /// <summary>
        /// Number of iterations in which this feature beat all shadow features.
        /// Normalised to [0, 1]. Higher = more consistently important.
        /// </summary>
        public float HitRatio { get; init; }

        /// <summary>Boruta-style classification of this feature.</summary>
        public FeatureImportanceVerdict Verdict { get; init; }
    }

}