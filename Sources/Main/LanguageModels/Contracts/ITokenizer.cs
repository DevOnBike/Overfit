namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public interface ITokenizer
    {
        int VocabularySize { get; }

        int EndOfTextTokenId { get; }

        int UnknownTokenId { get; }

        bool SupportsZeroAllocationEncode { get; }

        bool SupportsZeroAllocationDecode { get; }

        int CountTokens(ReadOnlySpan<char> text);

        int Encode(ReadOnlySpan<char> text, Span<int> destination);

        int Decode(ReadOnlySpan<int> tokens, Span<char> destination);

        string DecodeToString(ReadOnlySpan<int> tokens);
    }
}
