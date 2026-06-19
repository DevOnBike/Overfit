// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public interface ITokenizer
    {
        int VocabularySize
        {
            get;
        }

        int EndOfTextTokenId
        {
            get;
        }

        int UnknownTokenId
        {
            get;
        }

        bool SupportsZeroAllocationEncode
        {
            get;
        }

        bool SupportsZeroAllocationDecode
        {
            get;
        }

        int CountTokens(ReadOnlySpan<char> text);

        int Encode(ReadOnlySpan<char> text, Span<int> destination);

        int Decode(ReadOnlySpan<int> tokens, Span<char> destination);

        string DecodeToString(ReadOnlySpan<int> tokens);
    }
}
