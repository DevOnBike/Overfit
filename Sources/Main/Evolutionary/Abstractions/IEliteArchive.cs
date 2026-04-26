// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IEliteArchive : IDisposable
    {
        int DescriptorDimensions { get; }

        bool TryInsert(ReadOnlySpan<float> parameters, float fitness, ReadOnlySpan<float> descriptor);
    }
}