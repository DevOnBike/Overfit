// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com


// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Abstractions;

namespace DevOnBike.Overfit.Data.Features
{
    /// <summary>
    ///     Provider backed by a caller-supplied array of names.
    ///     Useful for datasets with known column names (e.g. loaded from CSV headers).
    /// </summary>
    public sealed class CustomFeatureNameProvider : IFeatureNameProvider
    {
        private readonly string[] _names;

        /// <param name="names">
        ///     One name per feature dimension. Length must match the training vector size.
        /// </param>
        public CustomFeatureNameProvider(string[] names)
        {
            if (names is null || names.Length == 0)
            {
                throw new ArgumentException("Names must not be null or empty.", nameof(names));
            }
            _names = names;
        }

        public int ExpectedDimension => _names.Length;

        public string GetName(int featureIndex)
        {
            return featureIndex < _names.Length
                ? _names[featureIndex] ?? $"feature_{featureIndex}"
                : $"feature_{featureIndex}";
        }
    }
}