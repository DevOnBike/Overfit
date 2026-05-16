// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Randomization;

namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    /// <summary>
    /// Selects a parent from the elite set.
    /// </summary>
    public interface ISelectionOperator
    {
        int SelectParent(ReadOnlySpan<int> eliteIndices, IRandom rng);
    }
}
