// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IEvolutionAlgorithm : IDisposable, IEvolutionCheckpoint
    {
        int PopulationSize { get; }
        int ParameterCount { get; }
        int Generation { get; }

        void Ask(Span<float> populationMatrix);   // [populationSize * parameterCount]
        void Tell(ReadOnlySpan<float> fitness);

        ReadOnlySpan<float> GetBestParameters();

        float BestFitness { get; }
    }
}