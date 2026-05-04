// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Adapter extension helpers for KeyValueCache.
    ///
    /// The first KV cache implementation predates the narrow read-only cache view.
    /// These helpers keep the next cached-attention code simple without changing
    /// the public IKeyValueCache contract yet.
    /// </summary>
    public static class KeyValueCacheReaderExtensions
    {
        public static IKeyValueCacheReader AsReader(this KeyValueCache cache)
        {
            if (cache is null)
            {
                throw new ArgumentNullException(nameof(cache));
            }

            return new KeyValueCacheReader(cache);
        }

        private sealed class KeyValueCacheReader : IKeyValueCacheReader
        {
            private readonly KeyValueCache _cache;

            public KeyValueCacheReader(KeyValueCache cache)
            {
                _cache = cache;
            }

            public int CurrentLength => _cache.CurrentLength;

            public int HeadDimension => _cache.Shape.HeadDimension;

            public ReadOnlySpan<float> GetKeyReadSpan(
                int layerIndex,
                int headIndex,
                int fromPosition,
                int length)
            {
                return _cache.GetKeyReadSpan(
                    layerIndex,
                    headIndex,
                    fromPosition,
                    length);
            }

            public ReadOnlySpan<float> GetValueReadSpan(
                int layerIndex,
                int headIndex,
                int fromPosition,
                int length)
            {
                return _cache.GetValueReadSpan(
                    layerIndex,
                    headIndex,
                    fromPosition,
                    length);
            }
        }
    }
}
