// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    public sealed class StopSequenceDetectorTests
    {
        [Fact]
        public void StopSplitAcrossPieces_EmitsPrefix_DropsStop()
        {
            var d = new StopSequenceDetector("STOP");

            Assert.Equal("abc", d.Append("abcST")); // holds "ST"
            Assert.Equal("", d.Append("OPxyz"));    // completes STOP
            Assert.True(d.Stopped);
            Assert.Equal("", d.Flush());
        }

        [Fact]
        public void PartialThatDoesNotComplete_IsEventuallyEmitted()
        {
            var d = new StopSequenceDetector("STOP");

            Assert.Equal("a", d.Append("aST"));   // holds "ST"
            Assert.Equal("STXb", d.Append("Xb")); // "ST" was not a stop after all
            Assert.False(d.Stopped);
            Assert.Equal("", d.Flush());
        }

        [Fact]
        public void NoStops_PassesThrough()
        {
            var d = new StopSequenceDetector();
            Assert.Equal("anything goes", d.Append("anything goes"));
            Assert.False(d.Stopped);
        }

        [Fact]
        public void MultipleStops_EarliestWins()
        {
            var d = new StopSequenceDetector("END", "STOP");
            Assert.Equal("x", d.Append("xENDySTOP"));
            Assert.True(d.Stopped);
        }

        [Fact]
        public void ExactStopInOnePiece_EmitsPrefix()
        {
            var d = new StopSequenceDetector("<|im_end|>");
            Assert.Equal("hello", d.Append("hello<|im_end|>"));
            Assert.True(d.Stopped);
        }

        [Fact]
        public void Flush_ReturnsHeldPartial_WhenStreamEndsWithoutStop()
        {
            var d = new StopSequenceDetector("STOP");
            Assert.Equal("ab", d.Append("abST")); // holds "ST"
            Assert.Equal("ST", d.Flush());        // dangling partial, never a stop
            Assert.False(d.Stopped);
        }

        [Fact]
        public void AppendAfterStopped_IsNoOp()
        {
            var d = new StopSequenceDetector("STOP");
            d.Append("STOP");
            Assert.True(d.Stopped);
            Assert.Equal("", d.Append("more text"));
        }
    }
}
