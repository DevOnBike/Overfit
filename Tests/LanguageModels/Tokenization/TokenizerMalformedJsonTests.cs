// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tokenization
{
    /// <summary>
    /// Pins the contract for a corrupt or hostile <c>tokenizer.json</c> — a file downloaded alongside every
    /// HuggingFace model, and parsed before any weights are touched.
    ///
    /// <para>The exposure here is not the JSON parser (<c>System.Text.Json</c> caps nesting depth on its own)
    /// but what is done with the <i>values</i>: the decoder table is sized from the largest token id the file
    /// declares. A vocabulary of one entry mapped to id 2 000 000 000 asks for a two-billion-element
    /// <c>string[]</c> — 16 GB of references — from a file of a few dozen bytes. The id is not a length, so
    /// nothing about the file's size bounds it; it has to be checked against the vocabulary it belongs to.</para>
    ///
    /// <para>Each case is run under a wall-clock budget so a regression fails the test rather than stalling
    /// the suite on an allocation the machine is large enough to attempt.</para>
    /// </summary>
    public sealed class TokenizerMalformedJsonTests : IDisposable
    {
        private static readonly TimeSpan LoadBudget = TimeSpan.FromSeconds(10);

        private readonly string _directory =
            Path.Combine(Path.GetTempPath(), "overfit-tokenizer-" + Guid.NewGuid().ToString("N"));

        public TokenizerMalformedJsonTests() => Directory.CreateDirectory(_directory);

        public void Dispose()
        {
            if (Directory.Exists(_directory))
            {
                Directory.Delete(_directory, recursive: true);
            }
        }

        private string WriteTokenizer(string json)
        {
            var path = Path.Combine(_directory, "tokenizer.json");
            File.WriteAllText(path, json, Encoding.UTF8);

            return path;
        }

        private static Exception? LoadWithinBudget(string path)
        {
            var work = Task.Run(() => Record.Exception(() => HuggingFaceBpeTokenizer.Load(path)));

            if (!work.Wait(LoadBudget))
            {
                Assert.Fail(
                    $"Loading did not finish within {LoadBudget.TotalSeconds:F0}s. A malformed tokenizer.json "
                    + "must be refused, not turned into a multi-gigabyte allocation.");
            }

            return work.Result;
        }

        [Fact]
        public void AbsurdTokenId_IsRefused_NotTurnedIntoADecoderTable()
        {
            // One real token plus one whose id is two billion. The decoder is sized from the maximum id, so
            // this is a request for a 16 GB string[] out of ~70 bytes of JSON.
            var path = WriteTokenizer(
                """{"model":{"type":"BPE","vocab":{"a":0,"b":2000000000},"merges":[]}}""");

            var thrown = LoadWithinBudget(path);

            Assert.NotNull(thrown);
            Assert.IsNotType<OutOfMemoryException>(thrown);
        }

        [Fact]
        public void TokenIdAtIntMaxValue_IsRefused_WithoutOverflowing()
        {
            // maxId + 1 overflows to int.MinValue, so the array length turns negative — a different failure
            // from the one above and worth its own case.
            var path = WriteTokenizer(
                """{"model":{"type":"BPE","vocab":{"a":0,"b":2147483647},"merges":[]}}""");

            var thrown = LoadWithinBudget(path);

            Assert.NotNull(thrown);
            Assert.IsNotType<OutOfMemoryException>(thrown);
        }

        [Fact]
        public void AbsurdAddedTokenId_IsRefused()
        {
            // The added-tokens list grows the same decoder table by a second route.
            var path = WriteTokenizer(
                """
                {"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},
                 "added_tokens":[{"id":2000000000,"content":"<|boom|>"}]}
                """);

            var thrown = LoadWithinBudget(path);

            Assert.NotNull(thrown);
            Assert.IsNotType<OutOfMemoryException>(thrown);
        }

        [Fact]
        public void TruncatedAndGarbageFiles_AreRefused_WithACatchableException()
        {
            var inputs = new[]
            {
                "",
                "{",
                """{"model":""",
                """{"model":{"type":"BPE"}}""",          // no vocab
                "this is not json at all",
            };

            foreach (var json in inputs)
            {
                var thrown = LoadWithinBudget(WriteTokenizer(json));

                Assert.NotNull(thrown);
                Assert.IsNotType<OutOfMemoryException>(thrown);
            }
        }

        [Fact]
        public void AWellFormedTinyVocabulary_IsStillAccepted()
        {
            // The guard must reject impossible ids without rejecting a legitimate small tokenizer — otherwise
            // it is not a validation, it is a different bug.
            var path = WriteTokenizer(
                """{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]},"added_tokens":[]}""");

            var thrown = LoadWithinBudget(path);

            Assert.Null(thrown);
        }
    }
}
