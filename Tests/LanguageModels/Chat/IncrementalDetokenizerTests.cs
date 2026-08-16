// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// <see cref="IncrementalDetokenizer"/> — the streaming rule that decides what a reader sees.
    ///
    /// <para><b>Why these tests did not exist before.</b> This logic lived inside a closure in
    /// <c>ChatSession.Generate</c>, reachable only by generating from a real model, so every test that
    /// covered it is a <c>[ModelFact]</c> that is skipped on any machine without the fixtures — including
    /// CI. The subtlest rule in the streaming path had no test at all. Extracting it was mostly about
    /// that; removing the per-token string allocation was the other half.</para>
    ///
    /// <para>The fake tokenizer below reports a scripted decode per step, which is the only way to
    /// reproduce the case that matters — a byte-level tokenizer RE-RENDERING earlier text once a following
    /// byte arrives.</para>
    /// </summary>
    public sealed class IncrementalDetokenizerTests
    {
        [Fact]
        public void EmitsOnlyWhatIsNew()
        {
            using var detokenizer = new IncrementalDetokenizer(initialCapacity: 16);
            var tokenizer = new ScriptedTokenizer(["He", "Hello", "Hello wo", "Hello world"]);

            Assert.Equal("He", Advance(detokenizer, tokenizer));
            Assert.Equal("llo", Advance(detokenizer, tokenizer));
            Assert.Equal(" wo", Advance(detokenizer, tokenizer));
            Assert.Equal("rld", Advance(detokenizer, tokenizer));
        }

        /// <summary>
        /// <b>The rule this class exists for.</b> A byte-level tokenizer renders a half-arrived codepoint
        /// as U+FFFD and then replaces it once the second byte lands. Emitting the delta against a prefix
        /// that changed underneath would send the reader a replacement character followed by the real
        /// letter — text the model never produced. The step must be held back instead.
        /// </summary>
        [Fact]
        public void AStepThatRewritesEarlierTextIsHeldBack()
        {
            using var detokenizer = new IncrementalDetokenizer(initialCapacity: 16);

            // "Cze" -> "Cze�" (half of 'ś' arrived) -> "Cześ" (it completed, so the FFFD is gone).
            var tokenizer = new ScriptedTokenizer(["Cze", "Cze�", "Cześ", "Cześć"]);

            Assert.Equal("Cze", Advance(detokenizer, tokenizer));
            Assert.Equal("�", Advance(detokenizer, tokenizer));

            // The prefix no longer matches what was emitted — nothing goes out this step.
            Assert.Null(Advance(detokenizer, tokenizer));

            // And it stays held back until the decode agrees with what the reader already has.
            Assert.Null(Advance(detokenizer, tokenizer));
        }

        /// <summary>A step that produces no new characters emits nothing rather than an empty delta.</summary>
        [Fact]
        public void AStepThatAddsNothingIsNotAdvanced()
        {
            using var detokenizer = new IncrementalDetokenizer(initialCapacity: 16);
            var tokenizer = new ScriptedTokenizer(["abc", "abc", "abcd"]);

            Assert.Equal("abc", Advance(detokenizer, tokenizer));
            Assert.Null(Advance(detokenizer, tokenizer));
            Assert.Equal("d", Advance(detokenizer, tokenizer));
        }

        /// <summary>
        /// The reply outgrows the initial buffer. Growth must carry the stabilised prefix with it — both
        /// buffers, not just the one being decoded into, or the next step compares against a truncated
        /// reference and holds back forever.
        /// </summary>
        [Fact]
        public void GrowingPastTheInitialCapacityKeepsStreaming()
        {
            using var detokenizer = new IncrementalDetokenizer(initialCapacity: 4);
            var steps = new string[64];
            var text = string.Empty;

            for (var i = 0; i < steps.Length; i++)
            {
                text += "abcdefgh";
                steps[i] = text;
            }

            var tokenizer = new ScriptedTokenizer(steps);
            var emitted = new System.Text.StringBuilder();

            for (var i = 0; i < steps.Length; i++)
            {
                emitted.Append(Advance(detokenizer, tokenizer));
            }

            var lastStep = steps[steps.Length - 1];

            Assert.Equal(lastStep, emitted.ToString());
            Assert.Equal(lastStep.Length, detokenizer.StableLength);
        }

        /// <summary>
        /// <b>The point of the change.</b> Zero bytes allocated per step once the pool is warm, when the
        /// tokenizer offers a span decode. The delta the caller turns into a string is not counted here —
        /// that one is linear in the reply and stays, because <c>StopSequenceDetector</c> takes a string.
        /// </summary>
        [Fact]
        public void AdvancingAllocatesNothingWhenTheTokenizerSupportsSpanDecode()
        {
            var steps = new string[256];
            var text = string.Empty;

            for (var i = 0; i < steps.Length; i++)
            {
                text += "xy";
                steps[i] = text;
            }

            // Warm the pool and the buffers at full size first, so this measures steady-state streaming
            // and not the growth that happens once per reply.
            for (var warm = 0; warm < 2; warm++)
            {
                using var warmUp = new IncrementalDetokenizer(initialCapacity: 1024);
                var warmTokenizer = new ScriptedTokenizer(steps);

                for (var i = 0; i < steps.Length; i++)
                {
                    warmTokenizer.Advance();
                    warmUp.TryAdvance(warmTokenizer, [1], out _);
                }
            }

            using var detokenizer = new IncrementalDetokenizer(initialCapacity: 1024);
            var tokenizer = new ScriptedTokenizer(steps);

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < steps.Length; i++)
            {
                tokenizer.Advance();
                detokenizer.TryAdvance(tokenizer, [1], out _);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0, allocated);
        }

        /// <summary>
        /// A tokenizer without a span decode still works — it goes through <c>DecodeToString</c> and a copy.
        /// One code path through the partial-codepoint rule, which is why the fallback lives here and not
        /// as a branch in <c>ChatSession</c>.
        /// </summary>
        [Fact]
        public void ATokenizerWithoutSpanDecodeStillStreams()
        {
            using var detokenizer = new IncrementalDetokenizer(initialCapacity: 4);
            var tokenizer = new ScriptedTokenizer(["ab", "abcd", "abcdef"]) { SpanDecode = false };

            Assert.Equal("ab", Advance(detokenizer, tokenizer));
            Assert.Equal("cd", Advance(detokenizer, tokenizer));
            Assert.Equal("ef", Advance(detokenizer, tokenizer));
        }

        [Fact]
        public void UsingItAfterDisposeThrows()
        {
            var detokenizer = new IncrementalDetokenizer();
            var tokenizer = new ScriptedTokenizer(["a"]);

            detokenizer.Dispose();

            Assert.Throws<ObjectDisposedException>(() => detokenizer.TryAdvance(tokenizer, [1], out _));
        }

        private static string? Advance(IncrementalDetokenizer detokenizer, ScriptedTokenizer tokenizer)
        {
            tokenizer.Advance();

            return detokenizer.TryAdvance(tokenizer, [1], out var delta) ? delta.ToString() : null;
        }

        /// <summary>
        /// Returns a scripted decode per step, so a re-render — the case a real tokenizer only produces on
        /// specific byte sequences — can be reproduced exactly.
        /// </summary>
        private sealed class ScriptedTokenizer : ITokenizer
        {
            private readonly string[] _steps;
            private int _step = -1;

            public ScriptedTokenizer(string[] steps) => _steps = steps;

            public bool SpanDecode { get; init; } = true;

            public void Advance() => _step++;

            public int VocabularySize => 8;

            public int EndOfTextTokenId => 0;

            public int UnknownTokenId => 0;

            public bool SupportsZeroAllocationEncode => false;

            public bool SupportsZeroAllocationDecode => SpanDecode;

            public int CountTokens(ReadOnlySpan<char> text) => text.Length;

            public int Encode(ReadOnlySpan<char> text, Span<int> destination) => 0;

            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
            {
                var current = _steps[_step];

                if (current.Length > destination.Length)
                {
                    throw new ArgumentException("too small", nameof(destination));
                }

                current.AsSpan().CopyTo(destination);

                return current.Length;
            }

            public string DecodeToString(ReadOnlySpan<int> tokens) => _steps[_step];
        }
    }
}
