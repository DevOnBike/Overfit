// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{

    public interface ISlmInferenceEngine : IDisposable
    {
        ISlmModel Model
        {
            get;
        }

        int VocabularySize
        {
            get;
        }

        int MaxContextLength
        {
            get;
        }

        bool SupportsKeyValueCache
        {
            get;
        }

        bool SupportsStreaming
        {
            get;
        }

        ISlmSession CreateSession();

        ISlmSession CreateSession(int maxContextLength);

        int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options);

        GenerationStats GenerateStreaming(
            ReadOnlySpan<int> promptTokens,
            in GenerationOptions options,
            TokenGeneratedHandler onToken);

        void ResetMetrics();

        GenerationStats GetLastGenerationStats();
    }
}
