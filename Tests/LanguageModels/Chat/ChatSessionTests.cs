// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// Loop/behaviour tests for <see cref="ChatSession"/> against fakes (no model):
    /// history accumulation, template-rendered prefill, token + string stops, streaming.
    /// </summary>
    public sealed class ChatSessionTests
    {
        private static readonly GenerationOptions Opts = new(
            maxNewTokens: 64, maxContextLength: 256, sampling: default, stopOnEndOfTextToken: true);

        [Fact]
        public void Send_AccumulatesHistory_AndPrefillsRenderedPrompt()
        {
            var session = new FakeSession("4");
            var tok = new CharTokenizer();
            var chat = new ChatSession(session, tok, new ChatTemplate(ChatTemplateFormat.ChatML));

            chat.AddSystem("sys");
            var reply = chat.Send("hi", in Opts);

            Assert.Equal("4", reply);
            // history: system, user, assistant
            Assert.Equal(3, chat.History.Count);
            Assert.Equal("assistant", chat.History[2].Role);
            Assert.Equal("4", chat.History[2].Content);

            // The session was prefilled with the ChatML render of system+user (+ assistant opener).
            var expectedPrompt = new ChatTemplate(ChatTemplateFormat.ChatML).Render(
                new[] { ChatMessage.System("sys"), ChatMessage.User("hi") });
            Assert.Equal(expectedPrompt, session.LastPromptText);
        }

        [Fact]
        public void Send_StopsOnEndOfTextToken()
        {
            // Emits 'a','b', then EOS — reply should be "ab".
            var session = new FakeSession("ab", appendEos: true);
            var chat = new ChatSession(session, new CharTokenizer(), new ChatTemplate(ChatTemplateFormat.ChatML));

            Assert.Equal("ab", chat.Send("x", in Opts));
        }

        [Fact]
        public void Send_TruncatesAtStringStop_AndStreams()
        {
            // Model rambles into a fake next turn; the string stop "\nUser:" cuts it.
            var session = new FakeSession("4\nUser: more");
            var chat = new ChatSession(
                session, new CharTokenizer(), new ChatTemplate(ChatTemplateFormat.ChatML),
                stopSequences: ["\nUser:"]);

            var streamed = new StringBuilder();
            var reply = chat.Send("q", in Opts, s => streamed.Append(s));

            Assert.Equal("4", reply);
            Assert.Equal("4", streamed.ToString()); // stop marker + tail never streamed
        }

        // ── Fakes ───────────────────────────────────────────────────────────
        // One token per char (token id = char code); EOS = -1.
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
                for (var i = 0; i < text.Length; i++) { destination[i] = text[i]; }
                return text.Length;
            }

            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
            {
                for (var i = 0; i < tokens.Length; i++) { destination[i] = (char)tokens[i]; }
                return tokens.Length;
            }

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder(tokens.Length);
                foreach (var t in tokens) { sb.Append((char)t); }
                return sb.ToString();
            }
        }

        private sealed class FakeSession : ISlmSession
        {
            private readonly Queue<int> _tokens = new();

            public FakeSession(string output, bool appendEos = false)
            {
                foreach (var c in output) { _tokens.Enqueue(c); }
                if (appendEos) { _tokens.Enqueue(-1); }
            }

            public string LastPromptText { get; private set; } = string.Empty;

            public int CurrentPosition { get; private set; }
            public int MaxContextLength => 4096;
            public int VocabularySize => 0x10000;
            public bool HasKeyValueCache => true;

            public void Reset() { }

            public void Reset(ReadOnlySpan<int> promptTokens)
            {
                var sb = new StringBuilder(promptTokens.Length);
                foreach (var t in promptTokens) { sb.Append((char)t); }
                LastPromptText = sb.ToString();
                CurrentPosition = promptTokens.Length;
            }

            public int GenerateNextToken(in SamplingOptions sampling)
            {
                CurrentPosition++;
                return _tokens.Count > 0 ? _tokens.Dequeue() : -1;
            }

            public int Generate(ReadOnlySpan<int> promptTokens, Span<int> outputTokens, in GenerationOptions options)
                => throw new NotSupportedException();

            public void GetLastLogits(Span<float> destination) => throw new NotSupportedException();

            public void Dispose() { }
        }
    }
}
