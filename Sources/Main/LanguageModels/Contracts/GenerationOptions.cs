// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public readonly struct GenerationOptions
    {
        public GenerationOptions(
            int maxNewTokens,
            int maxContextLength,
            SamplingOptions sampling,
            bool stopOnEndOfTextToken = true,
            int endOfTextTokenId = -1)
        {
            MaxNewTokens = maxNewTokens;
            MaxContextLength = maxContextLength;
            Sampling = sampling;
            StopOnEndOfTextToken = stopOnEndOfTextToken;
            EndOfTextTokenId = endOfTextTokenId;
        }

        public int MaxNewTokens { get; }

        public int MaxContextLength { get; }

        public SamplingOptions Sampling { get; }

        public bool StopOnEndOfTextToken { get; }

        public int EndOfTextTokenId { get; }
    }
}
