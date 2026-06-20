// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// An immutable snapshot of a <see cref="KeyValueCache"/>'s live region — the K/V of the first
    /// <see cref="Length"/> positions across every (layer, head). Used for <b>prefix / system-prompt KV
    /// reuse</b>: prefill a fixed prefix once, snapshot it, then restore it into a session for each new
    /// conversation/request (a memcpy) instead of re-encoding the prefix through the model every time.
    /// Stored compactly (only the prefix region, not the full cache).
    /// </summary>
    public sealed class KvCacheSnapshot
    {
        internal float[] Keys
        {
            get;
        }
        internal float[] Values
        {
            get;
        }
        internal int BasePosition
        {
            get;
        }
        internal int LayerCount
        {
            get;
        }
        internal int KvHeadCount
        {
            get;
        }
        internal int HeadDimension
        {
            get;
        }

        /// <summary>Number of cached positions (the prefix length).</summary>
        public int Length
        {
            get;
        }

        internal KvCacheSnapshot(
            float[] keys, float[] values, int length, int basePosition,
            int layerCount, int kvHeadCount, int headDimension)
        {
            Keys = keys;
            Values = values;
            Length = length;
            BasePosition = basePosition;
            LayerCount = layerCount;
            KvHeadCount = kvHeadCount;
            HeadDimension = headDimension;
        }

        internal bool MatchesShape(in KeyValueCacheShape shape)
            => shape.LayerCount == LayerCount
            && shape.KvHeadCount == KvHeadCount
            && shape.HeadDimension == HeadDimension;
    }
}
