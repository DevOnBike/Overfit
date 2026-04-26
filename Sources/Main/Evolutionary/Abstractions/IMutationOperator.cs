// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Mutates a parent genome into a child genome without allocating.
    /// </summary>
    public interface IMutationOperator
    {
        void Mutate(ReadOnlySpan<float> parentGenome, Span<float> childGenome, Random rng);
    }
}