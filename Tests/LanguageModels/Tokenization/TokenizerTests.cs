// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tokenization;
using Xunit;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// Tests for CharacterTokenizer, BytePairEncoder, and GPT-1 end-to-end pipeline.
    /// </summary>
    public class TokenizerTests
    {
        // ─────────────────────────────────────────────────────────────────────
        // CharacterTokenizer
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void CharTokenizer_FromCorpus_VocabContainsAllChars()
        {
            var tok = CharacterTokenizer.FromCorpus("hello world");

            // Unique chars: h, e, l, o, ' ', w, r, d = 8 + 3 special = 11
            Assert.Equal(11, tok.VocabSize);
        }

        [Fact]
        public void CharTokenizer_Encode_Decode_RoundTrip()
        {
            var tok  = CharacterTokenizer.FromCorpus("hello world");
            var ids  = tok.Encode("hello");
            var text = tok.Decode(ids);

            Assert.Equal("hello", text);
        }

        [Fact]
        public void CharTokenizer_UnknownChar_ReturnsUnknownId()
        {
            var tok = CharacterTokenizer.FromCorpus("hello");
            var ids = tok.Encode("z");  // 'z' not in corpus

            Assert.Equal(CharacterTokenizer.UnknownId, ids[0]);
        }

        [Fact]
        public void CharTokenizer_EmptyString_ReturnsEmptyArray()
        {
            var tok = CharacterTokenizer.FromCorpus("hello");
            var ids = tok.Encode("");

            Assert.Empty(ids);
        }

        [Fact]
        public void CharTokenizer_Ascii_HasCorrectVocabSize()
        {
            var tok = CharacterTokenizer.Ascii();
            // 95 printable + newline + tab + 3 special = 100
            Assert.Equal(100, tok.VocabSize);
        }

        [Fact]
        public void CharTokenizer_Ascii_EncodesAsciiText()
        {
            var tok  = CharacterTokenizer.Ascii();
            var ids  = tok.Encode("Hello!");
            var text = tok.Decode(ids);

            Assert.Equal("Hello!", text);
        }

        [Fact]
        public void CharTokenizer_SaveLoad_RoundTrip()
        {
            var tok1 = CharacterTokenizer.FromCorpus("abcdef");
            var path = Path.GetTempFileName();

            try
            {
                tok1.Save(path);
                var tok2 = CharacterTokenizer.Load(path);

                var ids1 = tok1.Encode("abc");
                var ids2 = tok2.Encode("abc");

                Assert.Equal(ids1, ids2);
                Assert.Equal(tok1.Decode(ids1), tok2.Decode(ids2));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void CharTokenizer_AllIds_InRange()
        {
            var tok  = CharacterTokenizer.Ascii();
            var ids  = tok.Encode("The quick brown fox");

            foreach (var id in ids)
            {
                Assert.True(id >= 0 && id < tok.VocabSize,
                    $"Token id {id} out of range [0, {tok.VocabSize})");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // GPT-1 end-to-end: tokenizer → model → logits → decode
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void GPT1_EndToEnd_CharLevel_GeneratesTokens()
        {
            // Build a small corpus tokenizer
            const string corpus = "hello world the quick brown fox jumps over the lazy dog";
            var tokenizer = CharacterTokenizer.FromCorpus(corpus);

            // Config: vocab matches tokenizer
            var config = new GPT1Config
            {
                VocabSize     = tokenizer.VocabSize,
                ContextLength = 32,
                DModel        = 32,
                NHeads        = 2,
                NLayers       = 1,
                DFF           = 64,
                TieWeights    = false,
            };

            using var model = new GPT1Model(config);
            model.Eval();

            // Encode prompt
            var prompt    = "hello";
            var promptIds = tokenizer.Encode(prompt);

            // Generate 5 more tokens
            var generated = model.Generate(promptIds, maxNewTokens: 5);

            // All generated ids should be valid token ids
            Assert.Equal(5, generated.Length);
            foreach (var id in generated)
            {
                Assert.True(id >= 0 && id < tokenizer.VocabSize,
                    $"Generated token {id} out of vocab range [0, {tokenizer.VocabSize})");
            }

            // Decode
            var text = tokenizer.Decode(generated);
            Assert.NotNull(text);
        }

        [Fact]
        public void GPT1_EndToEnd_GenerateLogits_ShapeIsVocabSize()
        {
            var tokenizer = CharacterTokenizer.FromCorpus("abcdef ghij");
            var config    = new GPT1Config
            {
                VocabSize     = tokenizer.VocabSize,
                ContextLength = 16,
                DModel        = 16,
                NHeads        = 2,
                NLayers       = 1,
                DFF           = 32,
                TieWeights    = false,
            };

            using var model   = new GPT1Model(config);
            model.Eval();

            var promptIds = tokenizer.Encode("abc");
            var logits    = model.GenerateLogits(promptIds);

            Assert.Equal(tokenizer.VocabSize, logits.Length);
            Assert.DoesNotContain(logits, float.IsNaN);
            Assert.DoesNotContain(logits, float.IsInfinity);
        }

        [Fact]
        public void GPT1_EndToEnd_DeterministicGreedy_SameResultTwice()
        {
            var tokenizer = CharacterTokenizer.Ascii();
            var config    = new GPT1Config
            {
                VocabSize     = tokenizer.VocabSize,
                ContextLength = 16,
                DModel        = 16,
                NHeads        = 2,
                NLayers       = 1,
                DFF           = 32,
                TieWeights    = false,
            };

            using var model = new GPT1Model(config);
            model.Eval();

            var prompt = tokenizer.Encode("Hi");
            var gen1   = model.Generate(prompt, maxNewTokens: 4);
            var gen2   = model.Generate(prompt, maxNewTokens: 4);

            Assert.Equal(gen1, gen2);
        }

        [Fact]
        public void GPT1_EndToEnd_ContextTruncation_WorksCorrectly()
        {
            var tok    = CharacterTokenizer.Ascii();
            var config = new GPT1Config
            {
                VocabSize     = tok.VocabSize,
                ContextLength = 8,  // very short context
                DModel        = 16,
                NHeads        = 2,
                NLayers       = 1,
                DFF           = 32,
                TieWeights    = false,
            };

            using var model = new GPT1Model(config);
            model.Eval();

            // 20-token prompt — model should truncate to last 8 tokens automatically
            var longPrompt = tok.Encode("Hello World Foo Bar");
            var generated  = model.Generate(longPrompt, maxNewTokens: 3);

            Assert.Equal(3, generated.Length);
        }
    }
}
