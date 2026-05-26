// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenization
{
    /// <summary>
    /// Synthetic tests for the SPM (SentencePiece) path of <see cref="GgufTokenizer"/> using a tiny
    /// hand-built vocab — validates the score-driven bigram merge, byte fallback, BOS, the
    /// <c>▁</c> space-prefix, and round-trip decode without needing a multi-GB GGUF file.
    /// </summary>
    public sealed class GgufTokenizerTests
    {
        // Vocab: 0 <unk>, 1 <s>, 2 </s>, 3..258 = <0x00>..<0xFF>, then normal pieces.
        private const int ByteBase = 3;
        private const int IdSpace = 259, IdA = 260, IdB = 261, IdAb = 262, IdSpaceA = 263;

        private static GgufTokenizer Build(bool addSpacePrefix = false)
        {
            var tokens = new string[264];
            var types = new int[264];
            var scores = new float[264];

            tokens[0] = "<unk>"; types[0] = 2;
            tokens[1] = "<s>"; types[1] = 3;
            tokens[2] = "</s>"; types[2] = 3;
            for (var bv = 0; bv < 256; bv++)
            {
                tokens[ByteBase + bv] = $"<0x{bv:X2}>";
                types[ByteBase + bv] = 6;   // BYTE
            }
            void Norm(int id, string s, float score) { tokens[id] = s; types[id] = 1; scores[id] = score; }
            Norm(IdSpace, "▁", -2f);
            Norm(IdA, "a", -2f);
            Norm(IdB, "b", -2f);
            Norm(IdAb, "ab", -1f);        // higher score than the singles ⇒ preferred merge
            Norm(IdSpaceA, "▁a", -3f);

            return GgufTokenizer.CreateForTest(tokens, types, scores, bos: 1, eos: 2, unk: 0,
                addBos: true, addSpacePrefix: addSpacePrefix);
        }

        [Fact]
        public void Merge_PrefersHigherScoreBigram_OverSingles()
        {
            var tok = Build();
            Assert.Equal(new[] { IdAb }, tok.Encode("ab", addBos: false));
        }

        [Fact]
        public void NoMergeAvailable_FallsBackToSingleChars()
        {
            var tok = Build();
            Assert.Equal(new[] { IdB, IdA }, tok.Encode("ba", addBos: false));   // "ba" not in vocab
        }

        [Fact]
        public void UnknownChar_FallsBackToByteTokens()
        {
            var tok = Build();
            // 'c' (0x63) is absent as a piece ⇒ one <0xNN> byte token.
            Assert.Equal(new[] { ByteBase + 0x63 }, tok.Encode("c", addBos: false));
        }

        [Fact]
        public void MultiByteChar_FallsBackToItsUtf8Bytes()
        {
            var tok = Build();
            // 'é' = U+00E9 = UTF-8 0xC3 0xA9 ⇒ two byte tokens, in order.
            Assert.Equal(new[] { ByteBase + 0xC3, ByteBase + 0xA9 }, tok.Encode("é", addBos: false));
        }

        [Fact]
        public void SpacePrefix_EscapesAndMerges()
        {
            var tok = Build(addSpacePrefix: true);
            // addSpacePrefix ⇒ "a" becomes "▁a", which merges to the single piece id 263.
            Assert.Equal(new[] { IdSpaceA }, tok.Encode("a", addBos: false));
        }

        [Fact]
        public void Bos_IsPrependedByDefault()
        {
            var tok = Build();
            Assert.Equal(new[] { 1, IdAb }, tok.Encode("ab"));   // addBos defaults to the file flag (true)
        }

        [Theory]
        [InlineData("ab")]
        [InlineData("ba")]
        [InlineData("c")]      // byte fallback
        [InlineData("é")]      // multi-byte fallback
        [InlineData("a")]
        public void RoundTrips(string text)
        {
            var tok = Build();
            Assert.Equal(text, tok.Decode(tok.Encode(text, addBos: false)));
        }

        [Fact]
        public void Decode_SkipsControlTokens()
        {
            var tok = Build();
            Assert.Equal("ab", tok.Decode(new[] { 1, IdAb, 2 }));   // <s> ab </s> → "ab"
        }

        // ── Real GGUF vocab (Mixtral SPM) ────────────────────────────────────

        [LongFact]
        public void RealMixtralVocab_RoundTrips()
        {
            const string path = @"C:\mixtral\mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf";
            if (!File.Exists(path)) { return; }

            var tok = GgufTokenizer.Load(path);
            Assert.Equal(32000, tok.VocabSize);

            foreach (var text in new[]
            {
                "The capital of France is Paris.",
                "Hello, world!",
                "def add(a, b): return a + b",
                "Café naïve résumé — über.",   // accented + em-dash (byte fallback)
            })
            {
                var ids = tok.Encode(text, addBos: false);
                Assert.Equal(text, tok.Decode(ids));
            }
        }
    }
}
