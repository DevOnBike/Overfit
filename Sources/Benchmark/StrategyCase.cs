// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace Benchmarks
{
    public sealed class StrategyCase
    {
        public StrategyCase(
            int populationSize,
            int parameterCount)
        {
            PopulationSize = populationSize;
            ParameterCount = parameterCount;
        }

        public int PopulationSize { get; }

        public int ParameterCount { get; }

        public override string ToString()
        {
            return $"P{PopulationSize}_N{ParameterCount}";
        }
    }
}