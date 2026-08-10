// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenizers
{
    /// <summary>
    /// The bound on decoder growth — <c>NR-8</c>, and the last
    /// <c>#pragma warning disable OVERFIT038</c> in the tree until it was replaced by this.
    ///
    /// <para><b>What it guards.</b> <c>QwenTokenizer.Load</c> grows its decoder to <c>id + 1</c> at 8 bytes
    /// an entry, and <c>id</c> comes out of <c>tokenizer.json</c> — a file somebody else wrote. A declared
    /// <c>int.MaxValue</c> is a 16 GB allocation request that the caller cannot catch: it is an
    /// <c>OutOfMemoryException</c> at best and the host's death at worst.</para>
    ///
    /// <para><b>Why the bound is the FILE'S length and not the token count.</b> The obvious bound,
    /// <c>vocab.Count + added.Count</c>, was measured against the real Qwen tokenizer before being rejected:
    /// it clears the highest added id by exactly ONE (151,664 against 151,665), so a tokenizer with a single
    /// gap in its added-token ids would be refused. Rejecting a valid model is worse than the allocation
    /// being guarded against. Bytes-in-the-file is the house pattern the binary loaders already use, and on
    /// that same file it clears by <b>11.6x</b> — the real file spends 46.4 bytes per token against the
    /// 4-byte floor assumed here.</para>
    ///
    /// <para>These tests write their own minimal <c>tokenizer.json</c>, so they run on every
    /// <c>dotnet test</c> and need no model fixture.</para>
    /// </summary>
    public sealed class QwenTokenizerDecoderBoundTests : IDisposable
    {
        private readonly string _dir = Path.Combine(
            Path.GetTempPath(), "overfit-tok-" + Guid.NewGuid().ToString("N"));

        public QwenTokenizerDecoderBoundTests()
        {
            Directory.CreateDirectory(_dir);
        }

        public void Dispose()
        {
            if (Directory.Exists(_dir))
            {
                Directory.Delete(_dir, recursive: true);
            }
        }

        /// <summary>
        /// A minimal but valid BPE tokenizer.json with one added token at <paramref name="addedId"/>, padded
        /// so the file has a realistic size. <paramref name="padTokens"/> controls the byte length, which is
        /// what the bound is computed from.
        /// </summary>
        private string WriteTokenizer(int addedId, int padTokens = 64)
        {
            var vocab = new StringBuilder();
            vocab.Append("\"a\":0,\"b\":1");

            for (var i = 0; i < padTokens; i++)
            {
                vocab.Append(",\"tok").Append(i).Append("\":").Append(i + 2);
            }

            var json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{" + vocab
                       + "},\"merges\":[\"a b\"]},\"added_tokens\":[{\"id\":" + addedId
                       + ",\"content\":\"<|special|>\"}]}";
            var path = Path.Combine(_dir, "tokenizer.json");
            File.WriteAllText(path, json, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            return path;
        }

        /// <summary>The defect: four bytes in a file asking for 16 GB of decoder.</summary>
        [Fact]
        public void AnIdThatCannotFitInTheFileIsRefused()
        {
            var path = WriteTokenizer(int.MaxValue);

            var error = Assert.Throws<OverfitFormatException>(() => QwenTokenizer.Load(path));

            Assert.Contains("2147483647", error.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// The message has to carry the arithmetic, not just the refusal: an operator holding a rejected
        /// model needs to see the declared id against what the file could possibly describe.
        /// </summary>
        [Fact]
        public void TheRefusalNamesTheFileLengthAndTheDerivedCap()
        {
            var path = WriteTokenizer(int.MaxValue);
            var size = new FileInfo(path).Length;

            var error = Assert.Throws<OverfitFormatException>(() => QwenTokenizer.Load(path));

            Assert.Contains(size.ToString(), error.Message, StringComparison.Ordinal);
            Assert.Contains((size / 4).ToString(), error.Message, StringComparison.Ordinal);
        }

        [Fact]
        public void ANegativeIdIsRefused()
        {
            var path = WriteTokenizer(-1);

            Assert.Throws<OverfitFormatException>(() => QwenTokenizer.Load(path));
        }

        /// <summary>
        /// <b>The side that matters more.</b> A guard that rejects valid models is worse than the allocation
        /// it prevents, so this pins that an ordinary added token — sitting above the vocabulary, exactly
        /// where a real tokenizer puts its specials — still loads.
        /// </summary>
        [Fact]
        public void AnOrdinaryAddedTokenAboveTheVocabularyStillLoads()
        {
            var path = WriteTokenizer(addedId: 66);

            var tokenizer = QwenTokenizer.Load(path);

            Assert.NotNull(tokenizer);
        }

        /// <summary>
        /// The case that killed the obvious bound: added-token ids need not be contiguous with the
        /// vocabulary. `vocab.Count + added.Count` would refuse this; a length-derived bound does not.
        /// </summary>
        [Fact]
        public void AnAddedTokenIdWithAGapAboveTheVocabularyStillLoads()
        {
            // The special sits far above the vocabulary rather than at count+1 — the shape that made
            // `vocab.Count + added.Count` unusable. The padding is what a real tokenizer carrying that many
            // ids would weigh: the bound is derived from bytes, so a file has to be big enough to describe
            // the ids it declares, and this test would be dishonest with a 1 KB fixture.
            var path = WriteTokenizer(addedId: 500, padTokens: 400);

            var tokenizer = QwenTokenizer.Load(path);

            Assert.NotNull(tokenizer);
        }

        /// <summary>
        /// The bound scales with the document, which is the whole point of deriving it from bytes: the same
        /// id is refused in a small file and accepted in a larger one.
        /// </summary>
        [Fact]
        public void TheSameIdIsRefusedInASmallFileAndAcceptedInALargerOne()
        {
            const int Id = 20_000;

            var small = WriteTokenizer(Id, padTokens: 8);
            Assert.Throws<OverfitFormatException>(() => QwenTokenizer.Load(small));

            var large = WriteTokenizer(Id, padTokens: 12_000);
            Assert.NotNull(QwenTokenizer.Load(large));
        }
    }
}
