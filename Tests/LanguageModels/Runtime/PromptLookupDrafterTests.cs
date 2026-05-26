// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>Unit tests for <see cref="PromptLookupDrafter"/> (n-gram speculative drafting).</summary>
    public sealed class PromptLookupDrafterTests
    {
        [Fact]
        public void Draft_ProposesTokensFollowingTheMatchedNgram()
        {
            // "1 2 3 1 2" → suffix "1 2" matched earlier at index 0; the tokens after it were "3 1 2".
            int[] history = [1, 2, 3, 1, 2];
            var draft = new int[4];
            var n = PromptLookupDrafter.Draft(history, draft, ngramMin: 1, ngramMax: 3);
            Assert.Equal(3, n);
            Assert.Equal([3, 1, 2], draft[..3]);   // the continuation after "1 2" (3 is the useful guess)
        }

        [Fact]
        public void Draft_PrefersLongerNgramMatch()
        {
            // suffix "x y z" occurs earlier; the longest match (3-gram) should win over shorter ones.
            int[] history = [9, 8, 7, 5, 1, 2, 3, 8, 7, 5];   // suffix "8 7 5" matched at idx 1 → next is "1"? no.
            // history: idx 1..3 = 8 7 5, followed by idx4=1. suffix (last 3) = idx7..9 = 8 7 5.
            var draft = new int[3];
            var n = PromptLookupDrafter.Draft(history, draft, ngramMin: 1, ngramMax: 3);
            Assert.Equal(3, n);
            Assert.Equal([1, 2, 3], draft[..3]);   // tokens that followed "8 7 5" last time
        }

        [Fact]
        public void Draft_PrefersMostRecentOccurrence_ForSameLength()
        {
            // suffix "5" occurs at idx 0 and idx 3; most recent (idx 3) wins → next token is idx4=2.
            int[] history = [5, 9, 9, 5, 2, 5];
            var draft = new int[1];
            var n = PromptLookupDrafter.Draft(history, draft, ngramMin: 1, ngramMax: 1);
            Assert.Equal(1, n);
            Assert.Equal(2, draft[0]);
        }

        [Fact]
        public void Draft_ReturnsZeroWhenNoMatch()
        {
            int[] history = [1, 2, 3, 4, 5];
            var draft = new int[4];
            Assert.Equal(0, PromptLookupDrafter.Draft(history, draft, ngramMin: 1, ngramMax: 3));
        }

        [Fact]
        public void Draft_BoundedByDraftBufferLength()
        {
            // "a b c d a b" → after "a b" comes "c d ..." but draft buffer holds only 1.
            int[] history = [1, 2, 3, 4, 1, 2];
            var draft = new int[1];
            var n = PromptLookupDrafter.Draft(history, draft, ngramMin: 2, ngramMax: 2);
            Assert.Equal(1, n);
            Assert.Equal(3, draft[0]);
        }

        [Fact]
        public void Draft_EmptyBufferOrTinyHistory_ReturnsZero()
        {
            Assert.Equal(0, PromptLookupDrafter.Draft([1, 2, 3], default, 1, 3));
            Assert.Equal(0, PromptLookupDrafter.Draft([7], new int[2], 1, 3));   // need a token before the suffix
        }
    }
}
