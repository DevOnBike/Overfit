// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Snac;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>The decode-side primitives of SNAC's residual VQ: the codebook gather (<c>decode_code</c>, with
    /// its (B,T,D)→(B,D,T) transpose and range check) and the time <c>repeat_interleave</c> that upsamples a
    /// lower-rate level before summation. Hand-computed, model-free.</summary>
    public sealed class SnacResidualVqTests
    {
        [Fact]
        public void DecodeCodebook_GathersRows_AndTransposes()
        {
            // codebook [size=3 × dim=2]: row0=(10,11) row1=(20,21) row2=(30,31)
            float[] table = [10f, 11f, 20f, 21f, 30f, 31f];
            int[] codes = [2, 0, 1]; // time=3
            var dst = new float[2 * 3]; // [dim × time]

            SnacResidualVq.DecodeCodebook(codes, table, dst, codebookSize: 3, dim: 2, time: 3);

            // Channel-major [dim × time]: dim0 across time = rows' element 0; dim1 = element 1.
            Assert.Equal([30f, 10f, 20f], dst[..3]);  // dim 0 over t=0,1,2  (codes 2,0,1)
            Assert.Equal([31f, 11f, 21f], dst[3..]);  // dim 1 over t=0,1,2
        }

        [Fact]
        public void DecodeCodebook_OutOfRangeCode_Throws()
        {
            float[] table = [1f, 2f];
            int[] codes = [5];
            var dst = new float[1];

            Assert.Throws<OverfitRuntimeException>(() =>
                SnacResidualVq.DecodeCodebook(codes, table, dst, codebookSize: 1, dim: 1, time: 1));
        }

        [Fact]
        public void RepeatInterleaveTime_ReplicatesEachFrame()
        {
            // [channels=2 × time=2], stride 3 → [2 × 6], each frame replicated 3×.
            float[] src = [1f, 2f, /* ch1 */ 3f, 4f];
            var dst = new float[2 * 6];

            SnacResidualVq.RepeatInterleaveTime(src, dst, channels: 2, time: 2, stride: 3);

            Assert.Equal([1f, 1f, 1f, 2f, 2f, 2f], dst[..6]);
            Assert.Equal([3f, 3f, 3f, 4f, 4f, 4f], dst[6..]);
        }

        [Fact]
        public void RepeatInterleaveTime_StrideOne_IsIdentity()
        {
            float[] src = [5f, 6f, 7f];
            var dst = new float[3];

            SnacResidualVq.RepeatInterleaveTime(src, dst, channels: 1, time: 3, stride: 1);

            Assert.Equal(src, dst);
        }

        [Fact]
        public void AveragePoolTime_MeansNonOverlappingWindows()
        {
            // [channels=2 × time=4], stride 2 → [2 × 2].
            float[] src = [1f, 2f, 3f, 4f, /* ch1 */ 10f, 20f, 30f, 40f];
            var dst = new float[2 * 2];

            SnacResidualVq.AveragePoolTime(src, dst, channels: 2, time: 4, stride: 2);

            Assert.Equal([1.5f, 3.5f], dst[..2]);   // (1+2)/2, (3+4)/2
            Assert.Equal([15f, 35f], dst[2..]);      // (10+20)/2, (30+40)/2
        }

        [Fact]
        public void EncodeCodebook_PicksNearestByCosine()
        {
            // codebook [size=3 × dim=2]: row0=(1,0) row1=(0,1) row2=(1,1)
            float[] codebook = [1f, 0f, 0f, 1f, 1f, 1f];
            // two frames (channel-major [dim=2 × time=2]): frame0≈(0.9,0.1)→row0, frame1≈(0.7,0.75)→row2
            float[] zE = [0.9f, 0.7f, /* dim1 */ 0.1f, 0.75f];
            var indices = new int[2];

            SnacResidualVq.EncodeCodebook(zE, dim: 2, time: 2, codebook, codebookSize: 3, indices);

            Assert.Equal(0, indices[0]);
            Assert.Equal(2, indices[1]);
        }

        [Fact]
        public void EncodeCodebook_ThenDecode_RoundTripsThroughTheChosenRow()
        {
            // Distinct directions so cosine is unambiguous: row0 +x, row1 +y, row2 −x.
            float[] codebook = [1f, 0f, 0f, 1f, -1f, 0f]; // 3 rows, dim 2
            float[] zE = [0.1f, /* dim1 */ 5f]; // strongly +y → row1
            var idx = new int[1];
            SnacResidualVq.EncodeCodebook(zE, 2, 1, codebook, 3, idx);

            var decoded = new float[2];
            SnacResidualVq.DecodeCodebook(idx, codebook, decoded, 3, 2, 1);

            Assert.Equal(1, idx[0]);
            Assert.Equal([0f, 1f], decoded);
        }
    }
}
