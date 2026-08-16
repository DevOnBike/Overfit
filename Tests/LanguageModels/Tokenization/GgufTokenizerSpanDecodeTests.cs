// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenization
{
    /// <summary>
    /// <c>GgufTokenizer.Decode(ids, Span&lt;char&gt;)</c> — the allocation-free decode, and its agreement
    /// with the string overload.
    ///
    /// <para><b>What this is for.</b> <c>ChatSession</c>'s incremental detokenizer re-decodes the WHOLE
    /// generated run once per token, so the string overload used to allocate a <c>StringBuilder</c>, a
    /// <c>List&lt;byte&gt;</c>, an array from <c>ToArray</c>, a string per flush and a string per token —
    /// per token, over a growing sequence — in an engine whose stated property is that decode allocates
    /// nothing. This is the span path that removes it, and the tests that stop the two overloads drifting
    /// apart.</para>
    ///
    /// <para>Built from the in-memory test factories, so these run on every <c>dotnet test</c> and need no
    /// model fixture. Both vocabularies are exercised: SentencePiece (byte fallback, space marker, the
    /// leading-space strip) and byte-level BPE (one raw byte per character).</para>
    /// </summary>
    public sealed class GgufTokenizerSpanDecodeTests
    {
        /// <summary>
        /// <b>The rule that matters most.</b> Two implementations of one decode drift silently: the span
        /// path is used in the hot loop and the string path in tests, so a divergence shows up as a
        /// mangled reply in production and a green suite. The leading-space strip is exactly where they
        /// nearly did diverge — it is applied to the assembled text, not per token.
        /// </summary>
        [Fact]
        public void TheSpanOverloadAgreesWithTheStringOverloadOnSentencePiece()
        {
            var tokenizer = Spm();

            foreach (var ids in SpmCases())
            {
                var expected = tokenizer.Decode(ids);

                Span<char> destination = new char[256];
                var written = tokenizer.Decode(ids, destination);

                Assert.Equal(expected, destination[..written].ToString());
            }
        }

        /// <summary>
        /// <b>Absolute expectations, and they exist because comparing the two overloads was not enough.</b>
        /// A mutation removing the space-marker substitution left every parity test green — both overloads
        /// share that code, so they agreed on the same wrong answer. Parity catches drift between
        /// implementations; only a known-good value catches a defect in what they share.
        /// </summary>
        [Theory]
        [InlineData(new[] { 3, 4 }, "Hello world")]
        [InlineData(new[] { 3 }, "Hello")]
        [InlineData(new[] { 5, 4 }, "test world")]
        [InlineData(new[] { ControlId, 3, 4, ControlId }, "Hello world")]
        public void SentencePieceDecodesToTheExpectedText(int[] ids, string expected)
        {
            var tokenizer = Spm();

            Span<char> destination = new char[64];
            var written = tokenizer.Decode(ids, destination);

            Assert.Equal(expected, destination[..written].ToString());
            Assert.Equal(expected, tokenizer.Decode(ids));
        }

        /// <summary>
        /// The byte-level side, likewise against a known value rather than against the other overload.
        /// <c>Ġ</c> is the GPT-2 alphabet's stand-in for a space and must come back as one.
        /// </summary>
        [Fact]
        public void ByteLevelBpeDecodesToTheExpectedText()
        {
            var tokenizer = Bpe();

            Span<char> destination = new char[64];
            var written = tokenizer.Decode([3, 4, 5], destination);

            Assert.Equal("Hello world!", destination[..written].ToString());
            Assert.Equal("Hello world!", tokenizer.Decode([3, 4, 5]));
        }

        [Fact]
        public void TheSpanOverloadAgreesWithTheStringOverloadOnByteLevelBpe()
        {
            var tokenizer = Bpe();

            foreach (var ids in BpeCases())
            {
                var expected = tokenizer.Decode(ids);

                Span<char> destination = new char[256];
                var written = tokenizer.Decode(ids, destination);

                Assert.Equal(expected, destination[..written].ToString());
            }
        }

        /// <summary>
        /// A multi-byte codepoint split across two byte-fallback tokens must come back as one character.
        /// Decoding each token separately yields replacement characters, which is why bytes are accumulated
        /// and flushed rather than converted per token — and it is the case a naive rewrite loses first.
        /// </summary>
        [Fact]
        public void ACodepointSplitAcrossByteTokensSurvives()
        {
            var tokenizer = Spm();

            // 'ą' is U+0105 -> 0xC4 0x85, arriving as two <0xNN> tokens.
            var ids = new[] { IdOfByte(0xC4), IdOfByte(0x85) };

            Span<char> destination = new char[16];
            var written = tokenizer.Decode(ids, destination);

            Assert.Equal("ą", destination[..written].ToString());
            Assert.Equal("ą", tokenizer.Decode(ids));
        }

        /// <summary>
        /// <b>The point of the whole change.</b> Zero bytes allocated per decode once the pool is warm.
        /// Measured on the thread rather than claimed: <c>GC.GetAllocatedBytesForCurrentThread</c> is exact
        /// and deterministic, unlike a timing benchmark, and allocation is what this change is about.
        /// </summary>
        [Fact]
        public void TheSpanOverloadAllocatesNothingOnceThePoolIsWarm()
        {
            var tokenizer = Spm();
            var ids = SpmCases()[0];
            Span<char> destination = new char[256];

            for (var warm = 0; warm < 32; warm++)
            {
                tokenizer.Decode(ids, destination);
            }

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 64; i++)
            {
                tokenizer.Decode(ids, destination);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            Assert.Equal(0, allocated);
        }

        /// <summary>
        /// <c>DecodeToken</c> used to be <c>Decode(new[] { id })</c> — an <c>int[1]</c> per call, on a
        /// method whose whole purpose is being called once per token. The returned string is unavoidable;
        /// the array was not.
        /// </summary>
        [Fact]
        public void DecodeTokenAllocatesOnlyItsResultString()
        {
            var tokenizer = Spm();
            const int Id = 5;

            for (var warm = 0; warm < 32; warm++)
            {
                tokenizer.DecodeToken(Id);
            }

            var expected = tokenizer.DecodeToken(Id);

            var before = GC.GetAllocatedBytesForCurrentThread();

            for (var i = 0; i < 64; i++)
            {
                tokenizer.DecodeToken(Id);
            }

            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            // 64 strings of this length and nothing else. The bound is deliberately generous on string
            // overhead and far below what one extra array per call would add.
            var stringBudget = 64 * (24 + (2 * Math.Max(expected.Length, 1)) + 8);

            Assert.True(allocated <= stringBudget,
                $"{allocated} B for 64 DecodeToken calls, over the {stringBudget} B its strings alone need.");
        }

        /// <summary>
        /// A destination that cannot hold the result is refused with the required length, not silently
        /// truncated — a shortened reply is indistinguishable from a model that stopped early.
        /// </summary>
        [Fact]
        public void ADestinationThatIsTooSmallIsRefusedRatherThanTruncated()
        {
            var tokenizer = Spm();
            var ids = SpmCases()[0];
            var needed = tokenizer.Decode(ids).Length;

            var error = Assert.Throws<ArgumentException>(() =>
            {
                Span<char> tooSmall = new char[needed - 1];
                tokenizer.Decode(ids, tooSmall);
            });

            Assert.Contains(needed.ToString(), error.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void AnEmptySequenceDecodesToEmpty()
        {
            var tokenizer = Spm();

            Span<char> destination = new char[8];

            Assert.Equal(0, tokenizer.Decode([], destination));
            Assert.Equal(string.Empty, tokenizer.Decode([]));
        }

        /// <summary>Control tokens are dropped by both overloads, user-defined ones kept literally.</summary>
        [Fact]
        public void ControlTokensAreDroppedAndUserDefinedTokensAreKept()
        {
            var tokenizer = Spm();
            var ids = new[] { ControlId, 5, UserDefinedId };

            Span<char> destination = new char[64];
            var written = tokenizer.Decode(ids, destination);
            var text = destination[..written].ToString();

            Assert.Equal(tokenizer.Decode(ids), text);
            Assert.DoesNotContain("<s>", text, StringComparison.Ordinal);
            Assert.Contains("<|tool|>", text, StringComparison.Ordinal);
        }

        // ── fixtures ────────────────────────────────────────────────────────────

        private const int ControlId = 1;
        private const int UserDefinedId = 2;
        private const int ByteBase = 6;

        /// <summary>
        /// A SentencePiece vocabulary: two specials, three word pieces carrying the space marker, then the
        /// 256 <c>&lt;0xNN&gt;</c> byte-fallback tokens.
        /// </summary>
        private static GgufTokenizer Spm()
        {
            var tokens = new string[ByteBase + 256];
            var types = new int[tokens.Length];

            tokens[0] = "<unk>";
            tokens[ControlId] = "<s>";
            types[ControlId] = 3;
            tokens[UserDefinedId] = "<|tool|>";
            types[UserDefinedId] = 4;
            tokens[3] = "▁Hello";
            tokens[4] = "▁world";
            tokens[5] = "▁test";

            for (var b = 0; b < 256; b++)
            {
                tokens[ByteBase + b] = $"<0x{b:X2}>";
                types[ByteBase + b] = 6;
            }

            return GgufTokenizer.CreateForTest(
                tokens, types, new float[tokens.Length],
                bos: ControlId, eos: ControlId, unk: 0, addBos: false, addSpacePrefix: true);
        }

        private static int IdOfByte(int value) => ByteBase + value;

        private static int[][] SpmCases() =>
        [
            [3, 4],
            [3, 4, 5],
            [5],
            [ControlId, 3, 4, ControlId],
            [IdOfByte(0xC4), IdOfByte(0x85), 4],
            [3, UserDefinedId, 4],
        ];

        private static GgufTokenizer Bpe()
        {
            // Byte-level: each char of a piece maps back to one raw byte through the GPT-2 alphabet, so
            // the pieces are written in that alphabet rather than as plain text.
            var tokens = new[] { "<unk>", "<s>", "<|tool|>", "Hello", "Ġworld", "!" };
            var types = new[] { 0, 3, 4, 0, 0, 0 };

            return GgufTokenizer.CreateBpeForTest(
                tokens, types, [], bos: 1, eos: 1, unk: 0, addBos: false, preType: "default");
        }

        private static int[][] BpeCases() =>
        [
            [3, 4],
            [3, 4, 5],
            [1, 3, 4, 1],
            [3, 2, 4],
        ];
    }
}
