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
