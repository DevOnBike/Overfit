// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{

    public interface IKeyValueCache : IDisposable
    {
        KeyValueCacheShape Shape
        {
            get;
        }

        int CurrentLength
        {
            get;
        }

        int MaxLength
        {
            get;
        }

        bool IsFull
        {
            get;
        }

        void Reset();

        void Advance(int tokenCount = 1);

        Span<float> GetKeyWriteSpan(int layerIndex, int headIndex, int position);

        Span<float> GetValueWriteSpan(int layerIndex, int headIndex, int position);

        ReadOnlySpan<float> GetKeyReadSpan(int layerIndex, int headIndex, int fromPosition, int length);

        ReadOnlySpan<float> GetValueReadSpan(int layerIndex, int headIndex, int fromPosition, int length);
    }
}
