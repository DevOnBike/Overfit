// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// Fast wiring tests for sliding-window in <see cref="ChatSession"/> (no model): with sliding
    /// enabled the chat loop keeps generating past the model's context length; without it, it stops
    /// at the limit; and requesting it on a session that doesn't support it fails fast. The actual
    /// eviction correctness is covered by <c>SlidingWindowTests</c> on a real model.
    /// </summary>
    public sealed class ChatSessionSlidingWindowTests
    {
        private static ChatTemplate ChatMl => new(ChatTemplateFormat.ChatML);

        [Fact]
        public void SlidingEnabled_GeneratesPastContextLength()
        {
            var session = new FakeSession(maxContext: 4, supportsSliding: true);
            var chat = new ChatSession(session, new FakeTokenizer(), ChatMl, stopSequences: null, slidingWindow: true);

            var options = new GenerationOptions(maxNewTokens: 10, maxContextLength: 4, sampling: SamplingOptions.Greedy);
            var reply = chat.Send("hi", in options);

            Assert.True(session.SlidingEnabledCalled, "ChatSession did not enable sliding on the session.");
            Assert.Equal(10, reply.Length);   // one 'x' per generated token — bounded by MaxNewTokens, not context
        }

        [Fact]
        public void SlidingDisabled_StopsAtContextLength()
        {
            var session = new FakeSession(maxContext: 4, supportsSliding: true);
            var chat = new ChatSession(session, new FakeTokenizer(), ChatMl);   // slidingWindow: false

            var options = new GenerationOptions(maxNewTokens: 10, maxContextLength: 4, sampling: SamplingOptions.Greedy);
            var reply = chat.Send("hi", in options);

            Assert.False(session.SlidingEnabledCalled);
            // Prompt fills 2 of 4 slots; the loop stops once CurrentPosition reaches the cap → 2 tokens.
            Assert.Equal(2, reply.Length);
        }

        [Fact]
        public void SlidingRequested_OnUnsupportedSession_ThrowsAtConstruction()
        {
            var session = new FakeSession(maxContext: 4, supportsSliding: false);
            Assert.Throws<NotSupportedException>(() =>
                new ChatSession(session, new FakeTokenizer(), ChatMl, stopSequences: null, slidingWindow: true));
        }

        // Counts generated tokens via CurrentPosition; never returns EOS so the loop runs to its bound.
        private sealed class FakeSession : ISlmSession
        {
            private readonly bool _supports;
            private int _pos;

            public FakeSession(int maxContext, bool supportsSliding)
            {
                MaxContextLength = maxContext;
                _supports = supportsSliding;
            }

            public bool SlidingEnabledCalled { get; private set; }

            public int CurrentPosition => _pos;
            public int MaxContextLength { get; }
            public int VocabularySize => 16;
            public bool HasKeyValueCache => true;
            public bool SupportsSlidingWindow => _supports;

            public void EnableSlidingWindow(int evictBlock = 0)
            {
                if (!_supports) { throw new NotSupportedException("fake: no sliding"); }
                SlidingEnabledCalled = true;
            }

            public void Reset() => _pos = 0;
            public void Reset(ReadOnlySpan<int> promptTokens) => _pos = promptTokens.Length;

            public int GenerateNextToken(in SamplingOptions sampling)
            {
                _pos++;
                return 7;   // arbitrary non-EOS token; FakeTokenizer decodes any token to "x"
            }

            public int Generate(ReadOnlySpan<int> promptTokens, Span<int> outputTokens, in GenerationOptions options)
                => throw new NotSupportedException();

            public void GetLastLogits(Span<float> destination) => throw new NotSupportedException();

            public void Dispose() { }
        }

        private sealed class FakeTokenizer : ITokenizer
        {
            public int VocabularySize => 16;
            public int EndOfTextTokenId => 0;
            public int UnknownTokenId => -1;
            public bool SupportsZeroAllocationEncode => true;
            public bool SupportsZeroAllocationDecode => true;

            public int CountTokens(ReadOnlySpan<char> text) => 2;   // prompt always tokenizes to 2 tokens

            public int Encode(ReadOnlySpan<char> text, Span<int> destination)
            {
                destination[0] = 1;
                destination[1] = 2;
                return 2;
            }

            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
            {
                for (var i = 0; i < tokens.Length; i++) { destination[i] = 'x'; }
                return tokens.Length;
            }

            public string DecodeToString(ReadOnlySpan<int> tokens) => new('x', tokens.Length);
        }
    }
}
