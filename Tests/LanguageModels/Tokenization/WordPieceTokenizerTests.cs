// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenization
{
    /// <summary>
    /// <see cref="WordPieceTokenizer"/>: the BERT BasicTokenizer + greedy longest-match WordPiece
    /// algorithm is verified against hand-built vocabularies (no model file needed); a [LongFact]
    /// exercises the real all-MiniLM-L6-v2 <c>vocab.txt</c>.
    /// </summary>
    public sealed class WordPieceTokenizerTests
    {
        // A small vocab covering the classic "unaffable" → un ##aff ##able example, plus specials.
        private static WordPieceTokenizer BuildToy()
        {
            var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
            var tokens = new[]
            {
                "[PAD]", "[UNK]", "[CLS]", "[SEP]",     // 0..3 specials
                "un", "##aff", "##able", "##b",          // 4..7
                "hello", "world", ",", "!", "the", "a",  // 8..13
                "em", "##bed", "##ding", "##s",          // 14..17
            };
            for (var i = 0; i < tokens.Length; i++)
            {
                vocab[tokens[i]] = i;
            }
            return new WordPieceTokenizer(vocab, doLowerCase: true);
        }

        [Fact]
        public void GreedyLongestMatch_SplitsIntoSubwords()
        {
            var tok = BuildToy();
            // addSpecialTokens:false isolates the subword logic from [CLS]/[SEP].
            var ids = tok.Encode("unaffable", addSpecialTokens: false);
            Assert.Equal(new[] { 4, 5, 6 }, ids); // un ##aff ##able
        }

        [Fact]
        public void MultipleWords_GreedyAndContinuation()
        {
            var tok = BuildToy();
            var ids = tok.Encode("embeddings", addSpecialTokens: false);
            Assert.Equal(new[] { 14, 15, 16, 17 }, ids); // em ##bed ##ding ##s
        }

        [Fact]
        public void Encode_WrapsInClsSep()
        {
            var tok = BuildToy();
            var ids = tok.Encode("hello world");
            Assert.Equal(new[] { 2, 8, 9, 3 }, ids); // [CLS] hello world [SEP]
        }

        [Fact]
        public void Punctuation_SplitsAsStandaloneTokens()
        {
            var tok = BuildToy();
            var ids = tok.Encode("hello, world!", addSpecialTokens: false);
            Assert.Equal(new[] { 8, 10, 9, 11 }, ids); // hello , world !
        }

        [Fact]
        public void Lowercasing_IsApplied()
        {
            var tok = BuildToy();
            Assert.Equal(tok.Encode("HELLO", addSpecialTokens: false), tok.Encode("hello", addSpecialTokens: false));
        }

        [Fact]
        public void UnknownWord_MapsToUnk()
        {
            var tok = BuildToy();
            var ids = tok.Encode("xyzzy", addSpecialTokens: false);
            Assert.Equal(new[] { 1 }, ids); // [UNK]
        }

        [Fact]
        public void PartialMatch_FallsBackToUnkForWholeWord()
        {
            var tok = BuildToy();
            // "unx": "un" matches but "##x" does not, so the whole word becomes [UNK] (BERT semantics).
            var ids = tok.Encode("unx", addSpecialTokens: false);
            Assert.Equal(new[] { 1 }, ids);
        }

        [Fact]
        public void AccentStripping_NormalizesDiacritics()
        {
            var tok = BuildToy();
            // "héllo" → strip accent → "hello".
            Assert.Equal(new[] { 8 }, tok.Encode("héllo", addSpecialTokens: false));
        }

        [Fact]
        public void MissingRequiredSpecialToken_Throws()
        {
            var vocab = new Dictionary<string, int>(StringComparer.Ordinal) { ["hello"] = 0 };
            Assert.Throws<OverfitFormatException>(() => new WordPieceTokenizer(vocab));
        }

        [Fact]
        public void RoundTrip_DecodeReconstructsWords()
        {
            var tok = BuildToy();
            var ids = tok.Encode("hello world");
            Assert.Equal("hello world", tok.DecodeToString(ids));
        }

        // ---- real model ----

        [LongFact]
        public void RealMiniLmVocab_LoadsAndTokenizes()
        {
            var tok = WordPieceTokenizer.FromVocabFile(TestModelPaths.MiniLm.RequireVocabPath());

            // bert-base-uncased / MiniLM vocab is 30522 tokens with the standard special ids.
            Assert.Equal(30522, tok.VocabularySize);
            Assert.Equal(101, tok.ClassifierTokenId);
            Assert.Equal(102, tok.SeparatorTokenId);
            Assert.Equal(100, tok.UnknownTokenId);
            Assert.Equal(0, tok.PaddingTokenId);

            // Known reference tokenization (HF bert-base-uncased):
            //   "Hello, world!" → [CLS] hello , world ! [SEP] = 101 7592 1010 2088 999 102
            var ids = tok.Encode("Hello, world!");
            Assert.Equal(new[] { 101, 7592, 1010, 2088, 999, 102 }, ids);

            // "embeddings" → em ##bed ##ding ##s = 7861 8270 4667 2015
            var emb = tok.Encode("embeddings", addSpecialTokens: false);
            Assert.Equal(new[] { 7861, 8270, 4667, 2015 }, emb);
        }
    }
}
