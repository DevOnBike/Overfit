// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IFitnessShaper
    {
        void Shape(ReadOnlySpan<float> rawFitness, Span<float> shapedFitness);
    }
}