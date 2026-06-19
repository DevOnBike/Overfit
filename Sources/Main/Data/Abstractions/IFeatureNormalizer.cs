// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Abstractions
{
    /// <summary>
    /// Unified interface for normalization algorithms (e.g. Z-Score, Min-Max, Log1p).
    /// </summary>
    public interface IFeatureNormalizer
    {
        bool IsFrozen
        {
            get;
        }

        /// <summary>
        /// Trains the normalizer on historical data (Offline).
        /// </summary>
        void FitBatch(ReadOnlySpan<float> data);

        /// <summary>
        /// Freezes the computed parameters (e.g. mean, variance, min/max) for use in production.
        /// </summary>
        void Freeze();

        /// <summary>
        /// Normalizes data in place. Requires a prior call to Freeze().
        /// </summary>
        void TransformInPlace(Span<float> data);

        /// <summary>
        /// Resets the algorithm state.
        /// </summary>
        void Reset();

        /// <summary>
        /// Saves the frozen parameters to a binary stream.
        /// </summary>
        void Save(BinaryWriter bw);

        /// <summary>
        /// Loads frozen parameters from a binary stream.
        /// </summary>
        void Load(BinaryReader br);
    }
}