// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints
{
    /// <summary>
    /// Tests the JSON-mode logit mask end-to-end against a tiny hand-built vocabulary: at each state
    /// only tokens that keep the document well-formed survive (others become -inf), and the
    /// end-of-text token is masked until the JSON is complete. Uses a fake tokenizer so it needs no
    /// model and stays in the fast suite.
    /// </summary>
    public sealed class JsonGrammarConstraintTests
    {
        // Vocab: structural tokens, a key, a value, whitespace, and an empty-text EOS.
        private static readonly string[] Vocab =
            ["{", "}", "\"", ":", "a", "1", "[", "]", " ", ",", ""];
        private const int OpenBrace = 0, CloseBrace = 1, Quote = 2, Colon = 3,
                          LetterA = 4, One = 5, OpenBracket = 6, CloseBracket = 7,
                          Space = 8, Comma = 9, Eos = 10;

        [Fact]
        public void AtRoot_OnlyValueStartTokensSurvive()
        {
            var c = new JsonGrammarConstraint(new FakeTokenizer(Vocab, Eos));
            var logits = MaskFreshZeros(c);

            Assert.True(Allowed(logits, OpenBrace));      // { starts an object
            Assert.True(Allowed(logits, OpenBracket));    // [ starts an array
            Assert.True(Allowed(logits, Quote));          // " starts a string
            Assert.True(Allowed(logits, One));            // 1 starts a number
            Assert.True(Allowed(logits, Space));          // leading whitespace is fine

            Assert.False(Allowed(logits, CloseBrace));    // } cannot start a document
            Assert.False(Allowed(logits, Colon));
            Assert.False(Allowed(logits, LetterA));       // bare letter is not a value
            Assert.False(Allowed(logits, Comma));
            Assert.False(Allowed(logits, Eos));           // not complete yet → no EOS
        }

        [Fact]
        public void AfterOpenBrace_OnlyKeyOrCloseSurvive()
        {
            var c = new JsonGrammarConstraint(new FakeTokenizer(Vocab, Eos));
            c.Accept(OpenBrace);
            var logits = MaskFreshZeros(c);

            Assert.True(Allowed(logits, Quote));          // "  → start of a key
            Assert.True(Allowed(logits, CloseBrace));     // } → empty object
            Assert.False(Allowed(logits, OpenBrace));     // value not legal before a key
            Assert.False(Allowed(logits, One));
            Assert.False(Allowed(logits, Eos));
        }

        [Fact]
        public void EmptyObject_IsComplete_AndAllowsEos()
        {
            var c = new JsonGrammarConstraint(new FakeTokenizer(Vocab, Eos));
            c.Accept(OpenBrace);
            c.Accept(CloseBrace);

            Assert.True(c.IsComplete);
            var logits = MaskFreshZeros(c);
            Assert.True(Allowed(logits, Eos));            // complete → EOS unmasked
            Assert.False(Allowed(logits, OpenBrace));     // nothing may follow a complete root value
        }

        [Fact]
        public void BuildsFullObject_CompleteOnlyAtEnd()
        {
            var c = new JsonGrammarConstraint(new FakeTokenizer(Vocab, Eos));
            // { "a" : 1 }
            foreach (var tok in new[] { OpenBrace, Quote, LetterA, Quote, Colon, One })
            {
                Assert.False(c.IsComplete);
                c.Accept(tok);
            }
            Assert.False(c.IsComplete);   // number at root is open until a delimiter
            c.Accept(CloseBrace);
            Assert.True(c.IsComplete);
        }

        private static float[] MaskFreshZeros(JsonGrammarConstraint c)
        {
            var logits = new float[Vocab.Length];
            c.ApplyMask(logits);
            return logits;
        }

        private static bool Allowed(float[] logits, int token) => !float.IsNegativeInfinity(logits[token]);

        private sealed class FakeTokenizer : ITokenizer
        {
            private readonly string[] _vocab;

            public FakeTokenizer(string[] vocab, int eos)
            {
                _vocab = vocab;
                EndOfTextTokenId = eos;
            }

            public int VocabularySize => _vocab.Length;
            public int EndOfTextTokenId { get; }
            public int UnknownTokenId => -1;
            public bool SupportsZeroAllocationEncode => false;
            public bool SupportsZeroAllocationDecode => false;

            public int CountTokens(ReadOnlySpan<char> text) => throw new NotSupportedException();
            public int Encode(ReadOnlySpan<char> text, Span<int> destination) => throw new NotSupportedException();
            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination) => throw new NotSupportedException();

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder();
                foreach (var t in tokens) { sb.Append(_vocab[t]); }
                return sb.ToString();
            }
        }
    }
}
