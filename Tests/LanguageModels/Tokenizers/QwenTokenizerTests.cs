// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Tokenizers;

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
        private static readonly string TokenizerPath = FindTokenizerJson();

        private static string FindTokenizerJson()
        {
            string[] candidates =
            [
                "test_fixtures/tokenizer/tokenizer.json",
                "test_fixtures/tokenizer.json",
            ];
            return candidates.FirstOrDefault(File.Exists) ?? candidates[0];
        }

        private QwenTokenizer? TryLoad()
        {
            if (!File.Exists(TokenizerPath))
            {
                Console.WriteLine($"SKIPPED: tokenizer.json not found at {TokenizerPath}");
                Console.WriteLine("Copy it from the HuggingFace cache:");
                Console.WriteLine(@"  %USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\*\tokenizer.json");
                return null;
            }

            return QwenTokenizer.Load(TokenizerPath);
        }

        [Fact]
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

        [Fact]
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

        [Fact]
        public void Encode_Decode_RoundTrip_SimpleAscii()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            const string input = "Hello, world! How are you?";
            var tokens  = tok.Encode(input);
            var decoded = tok.Decode(tokens);

            Assert.Equal(input, decoded);
            Console.WriteLine($"'{input}' → {tokens.Length} tokens → '{decoded}'");
        }

        [Fact]
        public void Encode_Decode_RoundTrip_Polish()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            const string input = "Cześć, jak się masz?";
            var tokens  = tok.Encode(input);
            var decoded = tok.Decode(tokens);

            Assert.Equal(input, decoded);
            Console.WriteLine($"'{input}' → {tokens.Length} tokens → '{decoded}'");
        }

        [Fact]
        public void Encode_SpecialTokens_Recognised()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            var tokens = tok.Encode("<|im_start|>user\nHello<|im_end|>");
            Assert.Contains(QwenTokenizer.ImStart, tokens);
            Assert.Contains(QwenTokenizer.ImEnd,   tokens);
            Console.WriteLine($"Special tokens: [{string.Join(", ", tokens)}]");
        }

        [Fact]
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

        [Fact]
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

        [Fact]
        public void DecodeToken_SingleToken_ReturnsString()
        {
            var tok = TryLoad();
            if (tok is null)
            {
                return;
            }

            // Encode a simple word and decode each token individually
            var tokens = tok.Encode("Hello");
            var sb     = new System.Text.StringBuilder();
            foreach (var t in tokens)
            {
                sb.Append(tok.DecodeToken(t));
            }

            Assert.Equal("Hello", sb.ToString());
        }
    }
}
