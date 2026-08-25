// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Core.Kernels
{
    /// <summary>
    /// The fused im2col convolution matches a naive direct convolution at the shapes where its work
    /// decomposition splits M, and the split arithmetic itself is pinned with exact integers.
    ///
    /// <para><b>Why this exists (`XC-92`).</b> The fused path dispatches one work item per 32-column
    /// N-panel, so a 14x14 output is seven items and at most seven workers can ever be busy. Measured
    /// 2026-08-25: those layers take 3.93 ms at one core, 0.87 at seven and 0.83 at sixteen. Splitting M
    /// two ways was measured to take them to 0.64, and turning that on by default is what this file
    /// guards.</para>
    ///
    /// <para><b>The oracle is a naive direct convolution</b> — six nested loops over the definition, not a
    /// second configuration of the same kernel. A decomposition that is self-consistently wrong cannot
    /// agree with it.</para>
    ///
    /// <para><b>What these tests do NOT cover, stated because a green run reads the same either way.</b>
    /// The split fires only when the pool has at least twice as many workers as the layer has panels, and
    /// the worker count comes from the machine. <see cref="Forward_MatchesNaiveConvolution"/>'s
    /// single-panel shape reaches it on any box with two or more processors; the seven-panel shape needs
    /// fourteen. The fused kernel itself needs AVX-512, so on a machine without it these run the unfused
    /// path and prove nothing about the split. <see cref="FusedMBlocksFor_IsTheMeasuredRule"/> has no such
    /// dependency and is the test that runs everywhere.</para>
    /// </summary>
    [Collection(ExclusiveProcessMeasurementCollection.Name)]
    public sealed class ConvGemmFusedMSplitTests
    {
        /// <summary>
        /// The split rule, driven with exact integers rather than through a dispatch whose worker count is
        /// whatever the box happens to have.
        ///
        /// <para>rowBlocks, nPanels, workers, expected. The cases are the decision's own boundaries: fewer
        /// panels than half the workers splits, more does not, and the split never asks for more blocks
        /// than there are row blocks to give.</para>
        /// </summary>
        public static TheoryData<int, int, int, int> SplitCases()
        {
            return new TheoryData<int, int, int, int>
            {
                // VGG-16's 14x14 convolutions: 7 panels, 64 row blocks. Two blocks at both pool sizes this
                // box runs — 16 workers pinned to physical cores, and the shipping 32 logical.
                { 64, 7, 16, 2 },
                { 64, 7, 32, 2 },

                // The old rule asked for ceil(workers/nPanels), which is 3 here and 5 at 32 workers. Both
                // measured no better than no split at all, so neither may come back by accident.
                { 64, 7, 15, 2 },

                // Exactly twice the panels is the edge the guard is written on: 14 workers over 7 panels
                // splits, 13 does not.
                { 64, 7, 14, 2 },
                { 64, 7, 13, 1 },

                // VGG's 28x28 convolutions produce 25 panels. Measured 2026-08-25 through the override
                // in `Conv2DGemmKernels.FusedMBlocksOverride`: two blocks is +4.7%, three +10.1% and four
                // +17.1% on those layers at 16 workers. Both pool sizes this box runs are here because the
                // 2026-08-18 figure the gate used to cite was taken at 32.
                //
                // These two cases add no mutation coverage the surrounding ones lack — removing the gate
                // reddens { 64, 7, 13, 1 } as well. They are here because they name the measured
                // configuration.
                { 64, 25, 32, 1 },
                { 64, 25, 16, 1 },

                // One row block cannot be split, whatever the panel count.
                { 1, 1, 32, 1 },

                // Fewer row blocks than the split wants: clamp, never produce an empty block.
                { 2, 1, 32, 2 },

                // A single worker, and the degenerate panel count a caller must not be able to divide by.
                { 64, 7, 1, 1 },
                { 64, 0, 32, 1 },
            };
        }

        [Theory]
        [MemberData(nameof(SplitCases))]
        public void FusedMBlocksFor_IsTheMeasuredRule(int rowBlocks, int nPanels, int workers, int expected)
        {
            Assert.Equal(expected, Conv2DGemmKernels.FusedMBlocksFor(rowBlocks, nPanels, workers));
        }

        /// <summary>
        /// Output values, at the two shapes whose panel count triggers the split.
        ///
        /// <para>inChannels, outChannels, spatial size, kernelSize, padding. The 5x5 shape produces 25
        /// output positions, one panel, so the split fires on any pool with two workers; the 14x14 shape is
        /// VGG-16's, seven panels, and needs fourteen.</para>
        /// </summary>
        public static TheoryData<int, int, int, int, int> ConvShapes()
        {
            return new TheoryData<int, int, int, int, int>
            {
                { 8, 32, 5, 3, 1 },
                { 8, 64, 14, 3, 1 },

                // A partial last row block (36 = 4*8 + 4) against a partial last panel, in one call.
                { 4, 36, 14, 3, 1 },

                // Wide enough that no split happens at all: 32x32 output is 32 panels.
                { 4, 32, 32, 3, 1 },
            };
        }

        [Theory]
        [MemberData(nameof(ConvShapes))]
        public void Forward_MatchesNaiveConvolution(
            int inChannels, int outChannels, int size, int kernelSize, int padding)
        {
            var input = Deterministic(inChannels * size * size, seed: 7);
            var kernels = Deterministic(outChannels * inChannels * kernelSize * kernelSize, seed: 13);
            var outSize = size + (2 * padding) - kernelSize + 1;
            var actual = new float[outChannels * outSize * outSize];

            Conv2DGemmKernels.Forward(
                input, kernels, ReadOnlySpan<float>.Empty, actual,
                batchSize: 1, inChannels, outChannels, size, size, kernelSize, padding, stride: 1);

            var k = inChannels * kernelSize * kernelSize;

            for (var oc = 0; oc < outChannels; oc++)
            {
                for (var oy = 0; oy < outSize; oy++)
                {
                    for (var ox = 0; ox < outSize; ox++)
                    {
                        var sum = 0f;
                        var absSum = 0f;

                        for (var ic = 0; ic < inChannels; ic++)
                        {
                            for (var ky = 0; ky < kernelSize; ky++)
                            {
                                var iy = oy + ky - padding;

                                if (iy < 0 || iy >= size)
                                {
                                    continue;
                                }

                                for (var kx = 0; kx < kernelSize; kx++)
                                {
                                    var ix = ox + kx - padding;

                                    if (ix < 0 || ix >= size)
                                    {
                                        continue;
                                    }

                                    var weight = kernels[
                                        (((oc * inChannels) + ic) * kernelSize * kernelSize)
                                        + (ky * kernelSize) + kx];
                                    var value = input[(ic * size * size) + (iy * size) + ix];

                                    sum += weight * value;
                                    absSum += MathF.Abs(weight * value);
                                }
                            }
                        }

                        // The same per-element bound ConvGemmMSplitTests carries, and for the same reason: a
                        // dot product of signed values cancels, so the result's magnitude is not the size of
                        // the arithmetic that produced it. gamma_K * sum|a*b| with gamma_K = K*u, doubled.
                        const float Unit = 5.9604645e-8f;

                        var index = (((oc * outSize) + oy) * outSize) + ox;
                        var bound = MathF.Max(1e-7f, 2f * k * Unit * absSum);

                        Assert.True(
                            MathF.Abs(actual[index] - sum) <= bound,
                            $"inC={inChannels} outC={outChannels} size={size}: "
                            + $"channel {oc} at ({oy},{ox}): kernel {actual[index]}, "
                            + $"naive reference {sum}, difference {MathF.Abs(actual[index] - sum)} "
                            + $"over a bound of {bound}");
                    }
                }
            }
        }

        /// <summary>
        /// The output is still right when the M-split is forced past what the shipping rule ever asks for.
        ///
        /// <para><b>Why this is not covered by the theory above.</b> The rule caps the split at two blocks
        /// and only where panels are scarce, so no test in this suite ever executes three or four. The row
        /// range is <c>rowBlocks * mb / mBlocks</c> to <c>rowBlocks * (mb + 1) / mBlocks</c>, which is
        /// exact for any divisor — but VGG's 64 row blocks are divisible by 2 and NOT by 3, so a rounding
        /// mistake is invisible at every block count the product uses.</para>
        ///
        /// <para>The shape is 28x28, which is 25 panels — the layer `XC-92` measured the override on.</para>
        /// </summary>
        [Theory]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(4)]
        public void Forward_MatchesNaiveConvolution_UnderTheForcedMBlockSplit(int blocks)
        {
            var previous = Conv2DGemmKernels.FusedMBlocksOverride;

            try
            {
                // Restores the previous value rather than a constant, so this cannot quietly cancel an
                // OVERFIT_CONV_FUSED_M_BLOCKS the run was started with.
                Conv2DGemmKernels.FusedMBlocksOverride = blocks;

                Forward_MatchesNaiveConvolution(
                    inChannels: 4, outChannels: 64, size: 28, kernelSize: 3, padding: 1);
            }
            finally
            {
                Conv2DGemmKernels.FusedMBlocksOverride = previous;
            }
        }

        /// <summary>
        /// Nothing is written past the output. The split indexes C by (row block, panel) through raw
        /// pointers, so an off-by-one in the M-block's row range lands outside the result and no value test
        /// on the result itself can see it.
        /// </summary>
        [Fact]
        public void Forward_WritesNothingPastTheResult()
        {
            const int InChannels = 8;
            const int OutChannels = 64;
            const int Size = 14;
            const int Guard = 64;
            const float Sentinel = -12345.5f;

            var input = Deterministic(InChannels * Size * Size, seed: 7);
            var kernels = Deterministic(OutChannels * InChannels * 9, seed: 13);
            var length = OutChannels * Size * Size;
            var buffer = new float[length + Guard];

            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = Sentinel;
            }

            Conv2DGemmKernels.Forward(
                input, kernels, ReadOnlySpan<float>.Empty, buffer.AsSpan(0, length),
                batchSize: 1, InChannels, OutChannels, Size, Size, kernelSize: 3, padding: 1, stride: 1);

            for (var i = length; i < buffer.Length; i++)
            {
                Assert.True(
                    buffer[i] == Sentinel,
                    $"the convolution wrote {buffer[i]} at index {i}, {i - length} elements past the result");
            }

            for (var i = 0; i < length; i++)
            {
                Assert.True(
                    buffer[i] != Sentinel,
                    $"output element {i} (channel {i / (Size * Size)}) was never written: "
                    + "the decomposition skipped it");
            }
        }

        private static float[] Deterministic(int length, int seed)
        {
            var values = new float[length];
            var state = (uint)(0x9E3779B9 + seed);

            for (var i = 0; i < length; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                values[i] = (((state & 0x00FFFFFF) / 16777216f) * 2f) - 1f;
            }

            return values;
        }
    }
}
