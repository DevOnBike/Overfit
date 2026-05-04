// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public interface ISlmSession : IDisposable
    {
        int CurrentPosition { get; }

        int MaxContextLength { get; }

        int VocabularySize { get; }

        bool HasKeyValueCache { get; }

        void Reset();

        void Reset(ReadOnlySpan<int> promptTokens);

        int GenerateNextToken(in SamplingOptions sampling);

        int Generate(
            ReadOnlySpan<int> promptTokens,
            Span<int> outputTokens,
            in GenerationOptions options);

        void GetLastLogits(Span<float> destination);
    }
}
