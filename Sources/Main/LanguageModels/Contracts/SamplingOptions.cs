// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public readonly struct SamplingOptions
    {
        public static SamplingOptions Greedy { get; } = new(
            strategy: SamplingStrategy.Greedy,
            temperature: 1.0f,
            topK: 0,
            topP: 1.0f,
            seed: 0);

        public SamplingOptions(
            SamplingStrategy strategy,
            float temperature,
            int topK,
            float topP,
            int seed)
        {
            Strategy = strategy;
            Temperature = temperature;
            TopK = topK;
            TopP = topP;
            Seed = seed;
        }

        public SamplingStrategy Strategy { get; }

        public float Temperature { get; }

        public int TopK { get; }

        public float TopP { get; }

        public int Seed { get; }
    }
}
