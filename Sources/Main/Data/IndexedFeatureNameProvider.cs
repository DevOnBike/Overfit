// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Abstractions;

namespace DevOnBike.Overfit.Data
{
    /// <summary>
    ///     Generic fallback provider that names features "feature_0", "feature_1" etc.
    ///     Use when domain-specific names are not available or not needed.
    /// </summary>
    public sealed class IndexedFeatureNameProvider : IFeatureNameProvider
    {

        private readonly string _prefix;

        /// <param name="expectedDimension">
        ///     Expected vector length. Pass 0 to skip validation.
        /// </param>
        /// <param name="prefix">Name prefix. Default: "feature".</param>
        public IndexedFeatureNameProvider(int expectedDimension = 0, string prefix = "feature")
        {
            ArgumentException.ThrowIfNullOrEmpty(prefix);

            ExpectedDimension = expectedDimension;
            _prefix = prefix;
        }
        public int ExpectedDimension { get; }

        public string GetName(int featureIndex)
        {
            return $"{_prefix}_{featureIndex}";
        }
    }
}