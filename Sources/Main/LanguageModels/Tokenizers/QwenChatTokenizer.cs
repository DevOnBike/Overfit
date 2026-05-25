// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// Adapts <see cref="QwenTokenizer"/> to the <see cref="ITokenizer"/> contract so it
    /// can drive the turnkey <see cref="DevOnBike.Overfit.LanguageModels.Chat.ChatSession"/>.
    /// <see cref="QwenTokenizer"/>'s BPE encode produces a variable-length <c>int[]</c>
    /// (no cheap pre-count), so the zero-allocation encode/decode capabilities are reported
    /// false and <see cref="CountTokens"/> performs a real encode — fine for chat-prompt
    /// sizes, which are tiny next to model inference.
    /// </summary>
    public sealed class QwenChatTokenizer : ITokenizer
    {
        private readonly QwenTokenizer _inner;

        public QwenChatTokenizer(QwenTokenizer inner)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        }

        public int VocabularySize => _inner.VocabSize;

        // Qwen has no dedicated unknown token; the BPE maps OOV pieces to <|endoftext|>.
        public int EndOfTextTokenId => QwenTokenizer.EndOfText;
        public int UnknownTokenId => QwenTokenizer.EndOfText;

        public bool SupportsZeroAllocationEncode => false;
        public bool SupportsZeroAllocationDecode => false;

        public int CountTokens(ReadOnlySpan<char> text) => _inner.Encode(new string(text)).Length;

        public int Encode(ReadOnlySpan<char> text, Span<int> destination)
        {
            var tokens = _inner.Encode(new string(text));
            if (destination.Length < tokens.Length)
            {
                throw new ArgumentException(
                    $"Destination ({destination.Length}) is smaller than the encoded token count ({tokens.Length}).",
                    nameof(destination));
            }
            tokens.CopyTo(destination);
            return tokens.Length;
        }

        public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
        {
            var text = _inner.Decode(tokens);
            if (destination.Length < text.Length)
            {
                throw new ArgumentException(
                    $"Destination ({destination.Length}) is smaller than the decoded char count ({text.Length}).",
                    nameof(destination));
            }
            text.AsSpan().CopyTo(destination);
            return text.Length;
        }

        public string DecodeToString(ReadOnlySpan<int> tokens) => _inner.Decode(tokens);
    }
}
