// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenizers
{
    /// <summary>
    /// Tests for QwenTokenizer.
    /// Requires tokenizer.json in test_fixtures/tokenizer/ — copy it from:
    ///   C:\Users\<user>\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\<hash>\tokenizer.json
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class QwenTokenizerTests
    {
        private static string TokenizerPath => TestModelPaths.Qwen3B.TokenizerJsonPath;

        /// <summary>
        /// Loads the Qwen tokenizer or throws <see cref="FileNotFoundException"/>
        /// (via <c>RequireTokenizerJsonPath</c>) when the fixture is missing.
        /// Returns the loaded tokenizer — never null, kept signature for source
        /// compatibility with callers that still check for null.
        /// </summary>
        private QwenTokenizer? TryLoad()
        {
            return QwenTokenizer.Load(TestModelPaths.Qwen3B.RequireTokenizerJsonPath());
        }

        [LongFact]
        public void Load_ValidFile_Succeeds()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            Assert.True(tok.VocabSize >= 150_000, $"Unexpected vocab size: {tok.VocabSize}");
            Console.WriteLine($"Vocab size: {tok.VocabSize}");
        }

        [LongFact]
        public void Encode_Hello_ReturnsNonEmptyTokens()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            var tokens = tok.Encode("Hello");
            Assert.NotEmpty(tokens);
            Assert.All(tokens, t => Assert.InRange(t, 0, tok.VocabSize - 1));
            Console.WriteLine($"'Hello' → [{string.Join(", ", tokens)}]");
        }

        [LongFact]
        public void Encode_Decode_RoundTrip_SimpleAscii()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            const string input = "Hello, world! How are you?";
            var tokens = tok.Encode(input);
            var decoded = tok.Decode(tokens);

            Assert.Equal(input, decoded);
            Console.WriteLine($"'{input}' → {tokens.Length} tokens → '{decoded}'");
        }

        [LongFact]
        public void Encode_Decode_RoundTrip_Polish()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            const string input = "Cześć, jak się masz?";
            var tokens = tok.Encode(input);
            var decoded = tok.Decode(tokens);

            Assert.Equal(input, decoded);
            Console.WriteLine($"'{input}' → {tokens.Length} tokens → '{decoded}'");
        }

        [LongFact]
        public void Encode_SpecialTokens_Recognised()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            var tokens = tok.Encode("<|im_start|>user\nHello<|im_end|>");
            Assert.Contains(QwenTokenizer.ImStart, tokens);
            Assert.Contains(QwenTokenizer.ImEnd, tokens);
            Console.WriteLine($"Special tokens: [{string.Join(", ", tokens)}]");
        }

        [LongFact]
        public void BosTokenId_Is151643()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            Assert.Equal(151643, QwenTokenizer.EndOfText);
            Assert.True(tok.IsSpecialToken(QwenTokenizer.EndOfText));
        }

        [LongFact]
        public void BuildChatPrompt_ContainsSystemAndUser()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            var tokens = tok.BuildChatPrompt("What is 2+2?");
            var decoded = tok.Decode(tokens);

            Assert.Contains("system", decoded);
            Assert.Contains("user", decoded);
            Assert.Contains("What is 2+2?", decoded);
            Assert.Contains("assistant", decoded);
            Console.WriteLine($"Chat prompt: {tokens.Length} tokens");
            Console.WriteLine(decoded);
        }

        [LongFact]
        public void DecodeToken_SingleToken_ReturnsString()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            // Encode a simple word and decode each token individually
            var tokens = tok.Encode("Hello");
            var sb = new StringBuilder();
            foreach (var t in tokens)
            {
                sb.Append(tok.DecodeToken(t));
            }

            Assert.Equal("Hello", sb.ToString());
        }
    }
}
