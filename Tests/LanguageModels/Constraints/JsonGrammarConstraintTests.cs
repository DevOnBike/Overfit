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
        public void RequireObject_AtRoot_OnlyOpenBraceOrWhitespaceSurvive()
        {
            var c = new JsonGrammarConstraint(new FakeTokenizer(Vocab, Eos), requireObject: true);
            var logits = MaskFreshZeros(c);

            Assert.True(Allowed(logits, OpenBrace));      // { is the only legal value start now
            Assert.True(Allowed(logits, Space));          // leading whitespace still fine

            Assert.False(Allowed(logits, OpenBracket));   // array root rejected
            Assert.False(Allowed(logits, Quote));         // string root rejected (the wrap-in-string case)
            Assert.False(Allowed(logits, One));           // number root rejected
            Assert.False(Allowed(logits, Eos));

            // Once the object has opened, the gate no longer applies — normal object rules resume.
            c.Accept(OpenBrace);
            var after = MaskFreshZeros(c);
            Assert.True(Allowed(after, Quote));           // a key string
            Assert.True(Allowed(after, CloseBrace));      // empty object
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

        /// <summary>
        /// A dead end must terminate, not produce a distribution the sampler cannot use.
        ///
        /// <para><b>Every logit went to negative infinity before this was fixed.</b> That is not a refusal:
        /// softmax over all <c>-inf</c> is NaN, so the sampler is handed a degenerate distribution and the
        /// caller gets whatever NaN comparisons happen to select. The sibling <c>JsonSchemaConstraint</c>
        /// deliberately unmasks end-of-text in exactly this case so generation stops on the valid prefix,
        /// and this class did not — one implementation of a pattern carrying the guard while its twin does
        /// not, which is the dominant defect shape in this codebase.</para>
        ///
        /// <para><b>Reaching a real dead end takes some care</b>, and the first attempt at this test did
        /// not: inside a JSON string almost every character is legal, so a vocabulary of structural tokens
        /// continues happily. After an opening brace, though, only whitespace, <c>"</c> or <c>}</c> may
        /// follow — a vocabulary holding none of those is genuinely stuck.</para>
        /// </summary>
        [Fact]
        public void ADeadEndUnmasksEndOfTextInsteadOfMaskingEverything()
        {
            // After '{' the machine wants a key quote or a closing brace. Neither exists here.
            string[] vocab = ["{", "a", ""];
            const int Eos2 = 2;

            var c = new JsonGrammarConstraint(new FakeTokenizer(vocab, Eos2));

            c.Accept(0);   // '{' — inside an object, expecting a key

            var logits = new float[vocab.Length];

            c.ApplyMask(logits);

            Assert.False(c.IsComplete);

            var survivors = 0;

            for (var t = 0; t < logits.Length; t++)
            {
                survivors += float.IsNegativeInfinity(logits[t]) ? 0 : 1;
            }

            Assert.True(
                survivors > 0,
                "every logit is -inf, so softmax over them is NaN — the sampler receives a degenerate "
                + "distribution rather than a stop signal");

            Assert.False(
                float.IsNegativeInfinity(logits[Eos2]),
                "the escape must be end-of-text, so generation ends on the valid prefix");
        }

        /// <summary>
        /// The escape must not fire while the document can still continue, or JSON mode stops being a
        /// guarantee — the model could end mid-object whenever it felt like it.
        /// </summary>
        [Fact]
        public void EndOfTextStaysMaskedWhileTheDocumentCanContinue()
        {
            var c = new JsonGrammarConstraint(new FakeTokenizer(Vocab, Eos));

            c.Accept(OpenBrace);

            var logits = new float[Vocab.Length];

            c.ApplyMask(logits);

            Assert.False(c.IsComplete);
            Assert.False(Allowed(logits, Eos));
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
            public int EndOfTextTokenId
            {
                get;
            }
            public int UnknownTokenId => -1;
            public bool SupportsZeroAllocationEncode => false;
            public bool SupportsZeroAllocationDecode => false;

            public int CountTokens(ReadOnlySpan<char> text) => throw new NotSupportedException();
            public int Encode(ReadOnlySpan<char> text, Span<int> destination) => throw new NotSupportedException();
            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination) => throw new NotSupportedException();

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder();
                foreach (var t in tokens)
                {
                    sb.Append(_vocab[t]);
                }
                return sb.ToString();
            }
        }
    }
}
