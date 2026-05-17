// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tokenization
{
    /// <summary>
    /// Minimal tokenizer contract for language model input/output.
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>Number of tokens in the vocabulary.</summary>
        int VocabSize { get; }

        /// <summary>Token id used to represent unknown tokens.</summary>
        int UnknownTokenId { get; }

        /// <summary>Encodes a string to a sequence of token ids.</summary>
        int[] Encode(string text);

        /// <summary>Decodes a sequence of token ids back to a string.</summary>
        string Decode(int[] tokenIds);

        /// <summary>Decodes a single token id to its string representation.</summary>
        string DecodeToken(int tokenId);
    }
}
