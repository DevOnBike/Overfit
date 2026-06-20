// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Threading.Tasks;
using DevOnBike.Overfit.Extensions.AI;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

namespace DevOnBike.Overfit.Tests.Adapters
{
    /// <summary>
    /// Fast tests (no model) for the <c>Microsoft.Extensions.AI</c> chat adapter: maps the M.E.AI
    /// <see cref="ChatMessage"/> list onto an Overfit <see cref="ChatSession"/>, streams deltas, reports
    /// usage + finish reason, and exposes <see cref="ChatClientMetadata"/>. Driven by a fake session.
    /// </summary>
    public sealed class OverfitChatClientTests
    {
        private static ChatSession NewSession(string reply)
            => new(new FakeSession(reply), new CharTokenizer(), new ChatTemplate(ChatTemplateFormat.ChatML));

        [Fact]
        public async Task GetResponseAsync_ReturnsAssistantMessage_WithUsage()
        {
            using var client = NewSession("Paris").AsChatClient("test-model");

            var response = await client.GetResponseAsync(
            [
                new ChatMessage(ChatRole.System, "be concise"),
                new ChatMessage(ChatRole.User, "capital of France?"),
            ]);

            Assert.Equal("Paris", response.Text);
            Assert.Equal(ChatRole.Assistant, response.Messages[^1].Role);
            Assert.Equal("test-model", response.ModelId);
            Assert.NotNull(response.Usage);
            Assert.Equal(5, response.Usage!.OutputTokenCount);   // "Paris" = 5 char-tokens
        }

        [Fact]
        public async Task GetStreamingResponseAsync_StreamsDeltas_ThatConcatToFullText()
        {
            using var client = NewSession("Paris").AsChatClient();

            var sb = new StringBuilder();
            ChatRole? firstRole = null;
            await foreach (var update in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "hi")]))
            {
                firstRole ??= update.Role;
                sb.Append(update.Text);
            }

            Assert.Equal(ChatRole.Assistant, firstRole);
            Assert.Equal("Paris", sb.ToString());
        }

        [Fact]
        public async Task MaxOutputTokens_TruncatesAndReportsLengthFinishReason()
        {
            using var client = NewSession("abcdef").AsChatClient();

            var response = await client.GetResponseAsync(
                [new ChatMessage(ChatRole.User, "go")],
                new ChatOptions { MaxOutputTokens = 3 });

            Assert.Equal("abc", response.Text);
            Assert.Equal(ChatFinishReason.Length, response.FinishReason);
        }

        [Fact]
        public void GetService_ReturnsMetadata()
        {
            using var client = NewSession("x").AsChatClient("m");

            var meta = client.GetService(typeof(ChatClientMetadata)) as ChatClientMetadata;

            Assert.NotNull(meta);
            Assert.Equal("overfit", meta!.ProviderName);
            Assert.Equal("m", meta.DefaultModelId);
        }

        // ── Fakes (same shape as ChatSessionTests; one token per char, EOS = -1) ──
        private sealed class CharTokenizer : ITokenizer
        {
            public int VocabularySize => 0x10000;
            public int EndOfTextTokenId => -1;
            public int UnknownTokenId => 0;
            public bool SupportsZeroAllocationEncode => true;
            public bool SupportsZeroAllocationDecode => true;

            public int CountTokens(ReadOnlySpan<char> text) => text.Length;

            public int Encode(ReadOnlySpan<char> text, Span<int> destination)
            {
                for (var i = 0; i < text.Length; i++)
                {
                    destination[i] = text[i];
                }
                return text.Length;
            }

            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
            {
                for (var i = 0; i < tokens.Length; i++)
                {
                    destination[i] = (char)tokens[i];
                }
                return tokens.Length;
            }

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder(tokens.Length);
                foreach (var t in tokens)
                {
                    sb.Append((char)t);
                }
                return sb.ToString();
            }
        }

        private sealed class FakeSession : ISlmSession
        {
            private readonly Queue<int> _tokens = new();

            public FakeSession(string output)
            {
                foreach (var c in output)
                {
                    _tokens.Enqueue(c);
                }
            }

            public int CurrentPosition
            {
                get; private set;
            }
            public int MaxContextLength => 4096;
            public int VocabularySize => 0x10000;
            public bool HasKeyValueCache => true;

            public void Reset()
            {
            }

            public void Reset(ReadOnlySpan<int> promptTokens) => CurrentPosition = promptTokens.Length;

            public int GenerateNextToken(in SamplingOptions sampling)
            {
                CurrentPosition++;
                return _tokens.Count > 0 ? _tokens.Dequeue() : -1;
            }

            public int Generate(ReadOnlySpan<int> promptTokens, Span<int> outputTokens, in GenerationOptions options)
                => throw new NotSupportedException();

            public void GetLastLogits(Span<float> destination) => throw new NotSupportedException();

            public void Dispose()
            {
            }
        }
    }
}
