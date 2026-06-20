// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Read-only view of a KV cache for cached attention decode.
    ///
    /// KeyValueCache implements the same methods through IKeyValueCache already.
    /// This interface exists to keep cached attention decoupled from the full
    /// mutable cache contract and make future specialized cache views possible.
    /// </summary>
    public interface IKeyValueCacheReader
    {
        int CurrentLength
        {
            get;
        }

        int HeadDimension
        {
            get;
        }

        ReadOnlySpan<float> GetKeyReadSpan(
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length);

        ReadOnlySpan<float> GetValueReadSpan(
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length);
    }
}
