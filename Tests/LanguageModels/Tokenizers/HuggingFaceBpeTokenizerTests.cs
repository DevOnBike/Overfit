// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenizers
{
    /// <summary>
    /// Validates the generic <see cref="HuggingFaceBpeTokenizer"/> (reads the pre-tokenizer
    /// Split regex + special tokens from <c>tokenizer.json</c>, no hard-coded per-model
    /// pattern) against Qwen's own files: round-trip identity, and parity with
    /// <see cref="QwenTokenizer"/> on non-numeric text, AND on digits (Qwen tokenizes one
    /// token per digit — the generic reader's file regex <c>\p{N}</c> and QwenTokenizer's
    /// baked <c>\p{N}{1,3}</c> give IDENTICAL output here because Qwen's merge table has no
    /// multi-digit merges; reading the regex from the file matters for OTHER families).
    /// [LongFact] — loads the real Qwen tokenizer.json.
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class HuggingFaceBpeTokenizerTests
    {
        private readonly ITestOutputHelper _out;
        public HuggingFaceBpeTokenizerTests(ITestOutputHelper output) => _out = output;

        private static HuggingFaceBpeTokenizer Load() =>
            HuggingFaceBpeTokenizer.Load(TestModelPaths.Qwen3B.RequireDir());

        [LongFact]
        public void Load_ResolvesVocabAndEos()
        {
            var tok = Load();
            Assert.True(tok.VocabularySize >= 150_000, $"vocab={tok.VocabularySize}");
            Assert.Equal(151643, tok.EndOfTextTokenId);   // <|endoftext|> from tokenizer_config.json
        }

        [LongFact]
        public void Encode_Decode_RoundTrips()
        {
            var tok = Load();
            string[] inputs =
            [
                "Hello, world! How are you?",
                "Cześć, jak się masz?",
                "The capital of France is Paris.",
                "emoji 🚀 and ünïcödé",
            ];
            foreach (var input in inputs)
            {
                var ids = tok.Encode(input);
                Assert.NotEmpty(ids);
                Assert.Equal(input, tok.DecodeToString(ids));
            }
        }

        [LongFact]
        public void SpecialTokens_Recognised()
        {
            var tok = Load();
            var ids = tok.Encode("<|im_start|>user\nHello<|im_end|>");
            Assert.Contains(QwenTokenizer.ImStart, ids);
            Assert.Contains(QwenTokenizer.ImEnd, ids);
        }

        [LongFact]
        public void MatchesQwenTokenizer_OnNonMultiDigitText()
        {
            var generic = Load();
            var qwen = QwenTokenizer.Load(TestModelPaths.Qwen3B.TokenizerJsonPath);

            string[] inputs =
            [
                "Hello, world! How are you?",
                "The capital of France is Paris.",
                "<|im_start|>user\nWhat is 2+2?<|im_end|>",   // single digits agree
            ];
            foreach (var input in inputs)
            {
                Assert.Equal(qwen.Encode(input), generic.Encode(input));
            }
        }

        [LongFact]
        public void Digits_TokenizeOneTokenPerDigit_AndAgreeWithQwen()
        {
            var generic = Load();
            var qwen = QwenTokenizer.Load(TestModelPaths.Qwen3B.TokenizerJsonPath);

            const string number = "1234567";
            var g = generic.Encode(number);
            _out.WriteLine($"'{number}' → {g.Length} tokens");

            // Qwen2.5 tokenizes digits one-per-token. NOTE: although the file regex is \p{N}
            // and QwenTokenizer's baked regex is \p{N}{1,3}, the output is IDENTICAL — Qwen's
            // merge table has no multi-digit merges, so any pre-tokenizer digit grouping is
            // split back to single digits by BPE. So the regex difference is immaterial here;
            // reading the regex from the file matters for OTHER families (different merges).
            Assert.Equal(number.Length, g.Length);                 // 7 digits → 7 tokens
            Assert.Equal(qwen.Encode(number), g);                  // full parity, even on digits
            Assert.Equal(number, generic.DecodeToString(g));       // round-trips
        }

        [LongFact]
        public void ITokenizerSurface_CountEncodeDecode_Consistent()
        {
            var tok = (DevOnBike.Overfit.LanguageModels.Contracts.ITokenizer)Load();
            const string input = "Hello, world!";

            var count = tok.CountTokens(input);
            var dst = new int[count];
            var written = tok.Encode(input, dst);

            Assert.Equal(count, written);
            Assert.Equal(input, tok.DecodeToString(dst));
        }
    }
}
