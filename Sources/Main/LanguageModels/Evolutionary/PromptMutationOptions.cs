// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Evolutionary
{
    public readonly struct PromptMutationOptions
    {
        public PromptMutationOptions(
            int maxPromptTokens,
            int maxGeneratedTokens,
            float temperature,
            int topK,
            int seed)
        {
            MaxPromptTokens = maxPromptTokens;
            MaxGeneratedTokens = maxGeneratedTokens;
            Temperature = temperature;
            TopK = topK;
            Seed = seed;
        }

        public int MaxPromptTokens
        {
            get;
        }

        public int MaxGeneratedTokens
        {
            get;
        }

        public float Temperature
        {
            get;
        }

        public int TopK
        {
            get;
        }

        public int Seed
        {
            get;
        }
    }
}
