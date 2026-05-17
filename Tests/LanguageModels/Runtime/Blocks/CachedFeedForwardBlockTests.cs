// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public class CachedFeedForwardBlockTests
    {
        [Fact]
        public void Constructor_ExposesShape()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 4,
                dFF: 8,
                activation: FeedForwardActivation.ReLU);

            Assert.Equal(4, block.DModel);
            Assert.Equal(8, block.DFF);
            Assert.Equal(FeedForwardActivation.ReLU, block.Activation);
        }

        [Fact]
        public void Constructor_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedFeedForwardBlock(
                    dModel: 0,
                    dFF: 8));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedFeedForwardBlock(
                    dModel: 4,
                    dFF: 0));
        }

        [Fact]
        public void Decode_NoneActivation_ComputesTwoLinearLayers()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 2,
                dFF: 3,
                activation: FeedForwardActivation.None);

            var hidden = new[] { 2f, 3f };

            // hidden [2] -> intermediate [3]
            // row0: [1, 2, 3]
            // row1: [4, 5, 6]
            var w1 = new[]
            {
                1f, 2f, 3f,
                4f, 5f, 6f
            };

            var b1 = new[] { 10f, 20f, 30f };

            // intermediate [3] -> output [2]
            // row0: [1, 2]
            // row1: [3, 4]
            // row2: [5, 6]
            var w2 = new[]
            {
                1f, 2f,
                3f, 4f,
                5f, 6f
            };

            var b2 = new[] { 100f, 200f };
            var output = new float[2];

            block.Decode(
                hidden,
                w1,
                b1,
                w2,
                b2,
                output);

            // intermediate:
            // [10 + 2*1 + 3*4, 20 + 2*2 + 3*5, 30 + 2*3 + 3*6]
            // [24, 39, 54]
            //
            // output:
            // out0 = 100 + 24*1 + 39*3 + 54*5 = 511
            // out1 = 200 + 24*2 + 39*4 + 54*6 = 728
            AssertClose(511f, output[0]);
            AssertClose(728f, output[1]);
        }

        [Fact]
        public void DecodeWithoutBias_UsesZeroBias()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 2,
                dFF: 2,
                activation: FeedForwardActivation.None);

            var hidden = new[] { 2f, 3f };

            var identity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var output = new float[2];

            block.DecodeWithoutBias(
                hidden,
                identity,
                identity,
                output);

            AssertClose(2f, output[0]);
            AssertClose(3f, output[1]);
        }

        [Fact]
        public void Decode_ReLU_ClampsNegativeIntermediate()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 2,
                dFF: 2,
                activation: FeedForwardActivation.ReLU);

            var hidden = new[] { 1f, 1f };

            var w1 = new[]
            {
                -1f, 2f,
                -1f, 3f
            };

            var w2 = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var output = new float[2];

            block.DecodeWithoutBias(
                hidden,
                w1,
                w2,
                output);

            AssertClose(0f, output[0]);
            AssertClose(5f, output[1]);
        }

        [Fact]
        public void Decode_GeLU_ProducesExpectedApproximation()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 1,
                dFF: 1,
                activation: FeedForwardActivation.GeLU);

            var hidden = new[] { 1f };
            var w1 = new[] { 1f };
            var w2 = new[] { 1f };
            var output = new float[1];

            block.DecodeWithoutBias(
                hidden,
                w1,
                w2,
                output);

            var expected = GeLU(1f);

            AssertClose(expected, output[0]);
        }

        [Fact]
        public void GetLastIntermediate_ReturnsActivatedIntermediate()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 2,
                dFF: 2,
                activation: FeedForwardActivation.ReLU);

            var hidden = new[] { 1f, 1f };

            var w1 = new[]
            {
                -1f, 2f,
                -1f, 3f
            };

            var identity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var output = new float[2];

            block.DecodeWithoutBias(
                hidden,
                w1,
                identity,
                output);

            var intermediate = new float[2];

            block.GetLastIntermediate(intermediate);

            Assert.Equal(new[] { 0f, 5f }, intermediate);
        }

        [Fact]
        public void GetLastIntermediate_DestinationTooSmall_Throws()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 2,
                dFF: 2);

            Assert.Throws<ArgumentException>(() =>
                block.GetLastIntermediate(new float[1]));
        }

        [Fact]
        public void Decode_InvalidArguments_Throw()
        {
            var block = new CachedFeedForwardBlock(
                dModel: 2,
                dFF: 3);

            var w1 = new float[2 * 3];
            var b1 = new float[3];
            var w2 = new float[3 * 2];
            var b2 = new float[2];
            var output = new float[2];

            Assert.Throws<ArgumentException>(() =>
                block.Decode(
                    hidden: new float[1],
                    w1,
                    b1,
                    w2,
                    b2,
                    output));

            Assert.Throws<ArgumentException>(() =>
                block.Decode(
                    hidden: new float[2],
                    w1: new float[5],
                    b1,
                    w2,
                    b2,
                    output));

            Assert.Throws<ArgumentException>(() =>
                block.Decode(
                    hidden: new float[2],
                    w1,
                    b1: new float[2],
                    w2,
                    b2,
                    output));

            Assert.Throws<ArgumentException>(() =>
                block.Decode(
                    hidden: new float[2],
                    w1,
                    b1,
                    w2: new float[5],
                    b2,
                    output));

            Assert.Throws<ArgumentException>(() =>
                block.Decode(
                    hidden: new float[2],
                    w1,
                    b1,
                    w2,
                    b2: new float[1],
                    output));

            Assert.Throws<ArgumentException>(() =>
                block.Decode(
                    hidden: new float[2],
                    w1,
                    b1,
                    w2,
                    b2,
                    output: new float[1]));
        }

        private static float GeLU(float x)
        {
            const float sqrtTwoOverPi = 0.7978845608028654f;
            const float coeff = 0.044715f;

            var x3 = x * x * x;
            var inner = sqrtTwoOverPi * (x + coeff * x3);

            return 0.5f * x * (1f + MathF.Tanh(inner));
        }

        private static void AssertClose(float expected, float actual)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= 1e-5f,
                $"Expected {expected}, actual {actual}.");
        }
    }
}
