// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Retrieval;

namespace DevOnBike.Overfit.Tests.LanguageModels.Retrieval
{
    /// <summary>
    /// Pins Reciprocal Rank Fusion. The property that matters is that <b>agreement across arms beats depth
    /// within one arm</b> — that is the entire reason hybrid retrieval works, and it is a property of the
    /// damping constant, so it needs a test rather than an assumption.
    /// </summary>
    public sealed class ReciprocalRankFusionTests
    {
        private static VectorMatch M(string id, string? payload = null) => new(id, 0f, payload);

        [Fact]
        public void DocumentInBothLists_BeatsDocumentRankedFirstInOnlyOne()
        {
            ReadOnlySpan<VectorMatch> dense = new[] { M("a"), M("b"), M("c") };
            ReadOnlySpan<VectorMatch> lexical = new[] { M("b"), M("d"), M("e") };

            var fused = ReciprocalRankFusion.Fuse(dense, lexical, 5);

            // "a" is rank 1 in dense but absent from lexical; "b" is rank 2 and rank 1 → agreement wins.
            Assert.Equal("b", fused[0].Id);
            Assert.Equal("a", fused[1].Id);
        }

        [Fact]
        public void ScoreIsTheSumOfReciprocalRanks()
        {
            ReadOnlySpan<VectorMatch> dense = new[] { M("a") };
            ReadOnlySpan<VectorMatch> lexical = new[] { M("x"), M("a") };

            var fused = ReciprocalRankFusion.Fuse(dense, lexical, 5);
            var a = fused[0];

            // rank 1 in dense + rank 2 in lexical, with the 1-based ranks of the RRF formula.
            var expected = (1f / (ReciprocalRankFusion.DefaultK + 1f)) + (1f / (ReciprocalRankFusion.DefaultK + 2f));
            Assert.Equal("a", a.Id);
            Assert.Equal(expected, a.Score, 6);
        }

        [Fact]
        public void SmallerK_SharpensTheAdvantageOfTopRanks()
        {
            ReadOnlySpan<VectorMatch> first = new[] { M("top"), M("second") };
            ReadOnlySpan<VectorMatch> second = [];

            var damped = ReciprocalRankFusion.Fuse(first, second, 2, k: 60f);
            var sharp = ReciprocalRankFusion.Fuse(first, second, 2, k: 1f);

            var dampedGap = damped[0].Score - damped[1].Score;
            var sharpGap = sharp[0].Score - sharp[1].Score;

            // This is exactly the knob that decides whether one confident arm can be outvoted.
            Assert.True(sharpGap > dampedGap, $"expected a sharper gap at k=1 ({sharpGap}) than k=60 ({dampedGap})");
        }

        [Fact]
        public void CarriesPayloadFromWhicheverArmSuppliedOne()
        {
            ReadOnlySpan<VectorMatch> dense = new[] { M("a", payload: null) };
            ReadOnlySpan<VectorMatch> lexical = new[] { M("a", payload: "chunk text") };

            var fused = ReciprocalRankFusion.Fuse(dense, lexical, 1);

            Assert.Equal("chunk text", fused[0].Payload);
        }

        [Fact]
        public void EmptyInputs_ProduceNoResults()
        {
            ReadOnlySpan<VectorMatch> none = [];
            Assert.Empty(ReciprocalRankFusion.Fuse(none, none, 5));
        }

        [Fact]
        public void OneEmptyArm_DegradesToTheOtherArmsOrder()
        {
            ReadOnlySpan<VectorMatch> dense = new[] { M("a"), M("b"), M("c") };
            ReadOnlySpan<VectorMatch> lexical = [];

            var fused = ReciprocalRankFusion.Fuse(dense, lexical, 3);

            Assert.Equal(["a", "b", "c"], fused.Select(m => m.Id));
        }

        [Fact]
        public void ResultsAreTruncatedToTheRequestedDepth()
        {
            ReadOnlySpan<VectorMatch> dense = new[] { M("a"), M("b"), M("c"), M("d") };
            ReadOnlySpan<VectorMatch> lexical = [];

            Span<VectorMatch> buffer = new VectorMatch[2];
            var written = ReciprocalRankFusion.Fuse(dense, lexical, buffer);

            Assert.Equal(2, written);
            Assert.Equal("a", buffer[0].Id);
            Assert.Equal("b", buffer[1].Id);
        }

        [Fact]
        public void RejectsNonPositiveK()
        {
            ReadOnlySpan<VectorMatch> none = [];
            Span<VectorMatch> buffer = new VectorMatch[1];

            // Span args cannot cross a lambda boundary, so the throwing call is made directly.
            var threw = false;
            try
            {
                ReciprocalRankFusion.Fuse(none, none, buffer, k: 0f);
            }
            catch (ArgumentOutOfRangeException)
            {
                threw = true;
            }

            Assert.True(threw, "expected ArgumentOutOfRangeException for k = 0");
        }
    }
}
