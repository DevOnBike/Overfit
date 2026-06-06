// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>Sentence splitting for chunked long-text synthesis (runs on already-normalized text). Model-free.</summary>
    public sealed class SentenceSplitterTests
    {
        [Fact]
        public void Split_ThreeSentences_KeepsPunctuation()
        {
            var s = SentenceSplitter.Split("Hello there. How are you? I am fine!");
            Assert.Equal(["Hello there.", "How are you?", "I am fine!"], s);
        }

        [Fact]
        public void Split_SingleSentence_OneChunk()
        {
            var s = SentenceSplitter.Split("Just one sentence here");
            Assert.Single(s);
            Assert.Equal("Just one sentence here", s[0]);
        }

        [Fact]
        public void Split_Empty_ReturnsNothing()
        {
            Assert.Empty(SentenceSplitter.Split("   "));
        }

        [Fact]
        public void Split_Ellipsis_NotOverSplit()
        {
            // Consecutive terminators collapse into one boundary.
            var s = SentenceSplitter.Split("Wait... really? Yes.");
            Assert.Equal(["Wait...", "really?", "Yes."], s);
        }
    }
}
