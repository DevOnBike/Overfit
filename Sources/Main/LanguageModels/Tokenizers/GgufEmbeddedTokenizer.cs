// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Tokenizers
{
    /// <summary>
    /// Adapts a <see cref="GgufTokenizer"/> (vocabulary reconstructed from the GGUF's embedded
    /// <c>tokenizer.ggml.*</c> metadata) to the <see cref="ITokenizer"/> surface, so a model can be
    /// tokenized from a bare <c>.gguf</c> with no side-loaded <c>tokenizer.json</c> / <c>vocab.json</c>.
    /// Used by <see cref="OverfitClient.LoadGguf"/> when the GGUF has no sibling tokenizer files.
    ///
    /// Encoding adds no BOS: the <see cref="DevOnBike.Overfit.LanguageModels.Chat.ChatTemplate"/>
    /// renders the model's special markers (e.g. ChatML <c>&lt;|im_start|&gt;</c>, Llama-3
    /// <c>&lt;|begin_of_text|&gt;</c>) into the prompt text, which the tokenizer maps to their special
    /// ids — so a tokenizer-injected BOS would double it. Encode/Decode allocate (the GGUF tokenizer
    /// works in <c>string</c>/<c>int[]</c>), so the zero-allocation flags are false; this is the prompt
    /// path, not the per-token decode hot path.
    /// </summary>
    public sealed class GgufEmbeddedTokenizer : ITokenizer
    {
        private readonly GgufTokenizer _inner;

        public GgufEmbeddedTokenizer(GgufTokenizer inner)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        }

        public int VocabularySize => _inner.VocabSize;
        public int EndOfTextTokenId => _inner.EosId;
        public int UnknownTokenId => _inner.UnknownId;
        public bool SupportsZeroAllocationEncode => false;
        public bool SupportsZeroAllocationDecode => false;

        public int CountTokens(ReadOnlySpan<char> text) => _inner.Encode(new string(text), addBos: false).Length;

        public int Encode(ReadOnlySpan<char> text, Span<int> destination)
        {
            var ids = _inner.Encode(new string(text), addBos: false);
            if (ids.Length > destination.Length)
            {
                throw new ArgumentException(
                    $"Destination ({destination.Length}) is smaller than the encoded length ({ids.Length}).",
                    nameof(destination));
            }
            ids.AsSpan().CopyTo(destination);
            return ids.Length;
        }

        public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
        {
            var text = _inner.Decode(tokens);
            if (text.Length > destination.Length)
            {
                throw new ArgumentException(
                    $"Destination ({destination.Length}) is smaller than the decoded length ({text.Length}).",
                    nameof(destination));
            }
            text.AsSpan().CopyTo(destination);
            return text.Length;
        }

        public string DecodeToString(ReadOnlySpan<int> tokens) => _inner.Decode(tokens);
    }
}
