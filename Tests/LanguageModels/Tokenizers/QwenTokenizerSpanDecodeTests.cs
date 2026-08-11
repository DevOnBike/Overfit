// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenizers
{
    /// <summary>
    /// <c>QwenTokenizer.Decode(tokens, Span&lt;char&gt;)</c> — the allocation-free decode and its agreement
    /// with the string overload.
    ///
    /// <para>The same change as on <c>GgufTokenizer</c>, for the same reason: <c>ChatSession</c> re-decodes
    /// the whole generated run once per token, and that used to cost a <c>List&lt;byte&gt;</c>, a
    /// <c>StringBuilder</c>, an array from <c>ToArray</c> and two strings <b>per token</b>.</para>
    ///
    /// <para><b>These write their own tokenizer.json</b> rather than taking a
    /// <c>[FixtureFact(TestFixture.QwenTokenizerJson)]</c>, so they run on every <c>dotnet test</c> and on
    /// CI. Every other test of this class needs the real 7 MB file and is therefore skipped almost
    /// everywhere — which is how a decode path ends up with no coverage that actually runs.</para>
    /// </summary>
    public sealed class QwenTokenizerSpanDecodeTests : IDisposable
    {
        private readonly string _directory = Path.Combine(
            Path.GetTempPath(), "overfit-qwen-span-" + Guid.NewGuid().ToString("N"));

        public QwenTokenizerSpanDecodeTests()
        {
            Directory.CreateDirectory(_directory);
        }

        public void Dispose()
        {
            if (Directory.Exists(_directory))
            {
                Directory.Delete(_directory, recursive: true);
            }
        }

        /// <summary>
        /// The two overloads must agree. They are two readings of one rule, one used in the hot loop and
        /// one in tests, so a divergence shows up as a mangled reply and a green suite.
        /// </summary>
        [Fact]
        public void TheSpanOverloadAgreesWithTheStringOverload()
        {
            var tokenizer = Build();

            foreach (var ids in Cases())
            {
                var expected = tokenizer.Decode(ids);

                Span<char> destination = new char[256];
                var written = tokenizer.Decode(ids, destination);

                Assert.Equal(expected, destination[..written].ToString());
            }
        }

        /// <summary>
        /// <b>Absolute values, not just parity.</b> Both overloads share the decode, so comparing them
        /// cannot catch a defect in what they share — a lesson from the GGUF side, where a mutation
        /// removing the space substitution left every parity test green.
        /// </summary>
        [Theory]
        [InlineData(new[] { 0, 1 }, "Hello world")]
        [InlineData(new[] { 0 }, "Hello")]
        [InlineData(new[] { 0, 1, 2 }, "Hello world!")]
        public void DecodesToTheExpectedText(int[] ids, string expected)
        {
            var tokenizer = Build();

            Span<char> destination = new char[64];
            var written = tokenizer.Decode(ids, destination);

            Assert.Equal(expected, destination[..written].ToString());
            Assert.Equal(expected, tokenizer.Decode(ids));
        }

        /// <summary>
        /// A special token interrupts the byte run: the accumulated bytes must be flushed BEFORE it is
        /// appended, or the text comes out in the wrong order — and the order is not visible in a
        /// single-token test.
        /// </summary>
        [Fact]
        public void ASpecialTokenFlushesTheBytesBeforeItself()
        {
            var tokenizer = Build();
            var ids = new[] { 0, SpecialId, 1 };

            Span<char> destination = new char[64];
            var written = tokenizer.Decode(ids, destination);

            Assert.Equal("Hello<|im_end|> world", destination[..written].ToString());
            Assert.Equal(tokenizer.Decode(ids), destination[..written].ToString());
        }

        /// <summary>
        /// A multi-byte codepoint spread over two byte-level pieces comes back as one character. Converting
        /// each piece separately would produce two replacement characters instead.
        /// </summary>
        [Fact]
        public void ACodepointSplitAcrossPiecesSurvives()
        {
            var tokenizer = Build();

            Span<char> destination = new char[16];
            var written = tokenizer.Decode([3, 4], destination);

            Assert.Equal("ą", destination[..written].ToString());
        }

        /// <summary><b>The point of the change:</b> zero bytes per decode once the pool is warm.</summary>
        [Fact]
        public void TheSpanOverloadAllocatesNothingOnceThePoolIsWarm()
        {
            var tokenizer = Build();
            var ids = new[] { 0, 1, 2 };
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

            Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        [Fact]
        public void ADestinationThatIsTooSmallIsRefusedRatherThanTruncated()
        {
            var tokenizer = Build();
            var ids = new[] { 0, 1 };
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
            var tokenizer = Build();

            Span<char> destination = new char[8];

            Assert.Equal(0, tokenizer.Decode([], destination));
            Assert.Equal(string.Empty, tokenizer.Decode([]));
        }

        // ── fixture ─────────────────────────────────────────────────────────────

        private const int SpecialId = 5;

        /// <summary>
        /// A minimal byte-level BPE vocabulary. Pieces are written in the GPT-2 alphabet, where printable
        /// ASCII maps to itself and a space is <c>Ġ</c> (U+0120); <c>Ä</c> (U+00C4) and <c>ħ</c> (U+0126)
        /// are that alphabet's stand-ins for the bytes 0xC4 and 0x85, which together are 'ą'.
        ///
        /// <para>Padded so the file is long enough for the added token's id to clear the decoder bound from
        /// <c>NR-8</c>, which is derived from the file's length at four bytes per token.</para>
        /// </summary>
        private QwenTokenizer Build()
        {
            var vocab = new StringBuilder();
            vocab.Append("\"Hello\":0,\"Ġworld\":1,\"!\":2,\"Ä\":3,\"ħ\":4");

            for (var i = 0; i < 400; i++)
            {
                vocab.Append(",\"pad").Append(i).Append("\":").Append(6 + i);
            }

            var json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{" + vocab
                       + "},\"merges\":[]},\"added_tokens\":[{\"id\":" + SpecialId
                       + ",\"content\":\"<|im_end|>\"}]}";

            var path = Path.Combine(_directory, "tokenizer.json");
            File.WriteAllText(path, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            return QwenTokenizer.Load(path);
        }

        private static int[][] Cases() =>
        [
            [0, 1],
            [0, 1, 2],
            [0, SpecialId, 1],
            [3, 4],
            [SpecialId],
            [],
        ];
    }
}
