// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.Core.Kernels
{
    /// <summary>
    /// Convolutions whose output does not divide evenly into the GEMM's 8x32 micro-tile.
    ///
    /// <para><b>Why this exists.</b> A full-tile fast path was added to the conv micro-kernel, and a mutation
    /// that routed <i>every</i> tile through it — including partial ones, where it stores eight full rows of
    /// thirty-two columns and therefore writes outside the valid region — <b>left all 2,728 tests green</b>.
    /// Nothing in the suite produced a partial tile, so both the fast path's guard and the general path's
    /// edge handling were unexercised.</para>
    ///
    /// <para>VGG-16 reaches partial tiles constantly: its later layers have <c>N = 784</c> and <c>N = 196</c>
    /// against a 32-column panel, which is 24.5 and 6.125 panels. The suite had no equivalent.</para>
    ///
    /// <para>The oracle is a direct convolution written here, not another configuration of the kernel under
    /// test, so a kernel that is self-consistently wrong cannot agree with it.</para>
    /// </summary>
    public sealed class ConvPartialTileTests
    {
        /// <summary>
        /// Shapes chosen against the micro-tile rather than against any real network: output channels that
        /// are not a multiple of 8, output positions that are not a multiple of 32, and both at once.
        /// </summary>
        public static TheoryData<int, int, int, int, int, int> Shapes()
        {
            return new TheoryData<int, int, int, int, int, int>
            {
                // inC, outC, H, W, kernel, padding — outC not a multiple of Mr=8, so the last row block is partial.
                { 3, 13, 9, 9, 3, 1 },

                // 7x7 output = 49 positions, not a multiple of Nr=32: the last column panel is partial.
                { 4, 16, 7, 7, 3, 1 },

                // Both edges at once, and outC below one row block entirely.
                { 5, 5, 11, 11, 3, 1 },

                // One full row block exactly, with a partial column tail.
                { 2, 8, 6, 10, 3, 1 },

                // Wide enough for several full panels plus a remainder of 4 columns.
                { 3, 9, 6, 34, 3, 1 },
            };
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void PartialTiles_MatchADirectConvolution(
            int inChannels,
            int outChannels,
            int inputH,
            int inputW,
            int kernelSize,
            int padding)
        {
            const int Stride = 1;
            const float Sentinel = -54321.5f;

            var outH = ((inputH + (2 * padding) - kernelSize) / Stride) + 1;
            var outW = ((inputW + (2 * padding) - kernelSize) / Stride) + 1;
            var outputSize = outChannels * outH * outW;

            var input = Deterministic(inChannels * inputH * inputW, seed: 2);
            var kernels = Deterministic(outChannels * inChannels * kernelSize * kernelSize, seed: 8);

            // A guard band, so a store that runs past the tile is caught rather than silently absorbed.
            const int Guard = 64;
            var output = new float[outputSize + Guard];

            for (var i = 0; i < output.Length; i++)
            {
                output[i] = Sentinel;
            }

            Conv2DKernels.ForwardNchw(
                input,
                kernels,
                output.AsSpan(0, outputSize),
                batchSize: 1,
                inChannels,
                outChannels,
                inputH,
                inputW,
                kernelSize,
                padding,
                Stride);

            var window = kernelSize * kernelSize;

            for (var oc = 0; oc < outChannels; oc++)
            {
                for (var oy = 0; oy < outH; oy++)
                {
                    for (var ox = 0; ox < outW; ox++)
                    {
                        var expected = 0f;
                        var absSum = 0f;

                        for (var ic = 0; ic < inChannels; ic++)
                        {
                            for (var ky = 0; ky < kernelSize; ky++)
                            {
                                var iy = (oy * Stride) - padding + ky;

                                if ((uint)iy >= (uint)inputH)
                                {
                                    continue;
                                }

                                for (var kx = 0; kx < kernelSize; kx++)
                                {
                                    var ix = (ox * Stride) - padding + kx;

                                    if ((uint)ix >= (uint)inputW)
                                    {
                                        continue;
                                    }

                                    var w = kernels[(oc * inChannels * window) + (ic * window) + (ky * kernelSize) + kx];
                                    var v = input[(ic * inputH * inputW) + (iy * inputW) + ix];

                                    expected += w * v;
                                    absSum += MathF.Abs(w * v);
                                }
                            }
                        }

                        var index = (oc * outH * outW) + (oy * outW) + ox;

                        // The textbook dot-product bound rather than a chosen tolerance: the two orders sum the
                        // same products differently, and the result cancels, so |expected| is not the size of
                        // the arithmetic behind it.
                        var terms = inChannels * window;
                        var bound = MathF.Max(1e-6f, 2f * terms * 5.9604645e-8f * absSum);

                        Assert.True(
                            MathF.Abs(output[index] - expected) <= bound,
                            $"inC={inChannels} outC={outChannels} {inputH}x{inputW} k={kernelSize}: "
                            + $"channel {oc}, position ({oy},{ox}): kernel {output[index]}, direct {expected}, "
                            + $"difference {MathF.Abs(output[index] - expected)} over bound {bound}");
                    }
                }
            }

            for (var i = outputSize; i < output.Length; i++)
            {
                Assert.True(
                    output[i] == Sentinel,
                    $"inC={inChannels} outC={outChannels}: the convolution wrote {output[i]} at index {i}, "
                    + $"{i - outputSize} elements past its output");
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
