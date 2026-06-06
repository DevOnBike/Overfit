// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Snac;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>The SNAC decoder's transposed-conv (learned-upsampling) primitive: hand-computed exact cases for
    /// the upsampling and channel-mixing math, the PyTorch output-length formula, and — the rigorous check — the
    /// fast gather kernel reproduced bit-for-bit by the canonical <i>scatter</i> definition across many shapes,
    /// including a size that crosses the parallel threshold. Model-free.</summary>
    public sealed class SnacConvTests
    {
        [Fact]
        public void StrideTwo_KernelOfOnes_DuplicatesInput()
        {
            // inC=outC=1, k=2, stride=2 with weight [1,1] is the classic nearest-neighbour x2 upsample.
            float[] input = [1f, 2f];
            float[] weight = [1f, 1f]; // [inC=1, outC=1, k=2]
            var tOut = SnacConv.OutputLength(tIn: 2, kSize: 2, stride: 2, pad: 0, dilation: 1, outputPadding: 0);
            var dst = new float[tOut];

            SnacConv.ConvTranspose1d(input, weight, ReadOnlySpan<float>.Empty, dst,
                inC: 1, tIn: 2, outC: 1, kSize: 2, stride: 2, pad: 0, dilation: 1, tOut: tOut);

            Assert.Equal(4, tOut);
            Assert.Equal([1f, 1f, 2f, 2f], dst);
        }

        [Fact]
        public void MultiInChannel_PointwiseMix_IsHandComputed()
        {
            // inC=2, outC=1, k=1, stride=1 → per-timestep linear mix. ch0 weight=2, ch1 weight=10.
            float[] input = [1f, 2f, /* ch1 */ 3f, 4f]; // [inC=2 × tIn=2], channel-major
            float[] weight = [2f, 10f];                  // [(ic*outC+oc)*k]: ic0->2, ic1->10
            var dst = new float[2];

            SnacConv.ConvTranspose1d(input, weight, ReadOnlySpan<float>.Empty, dst,
                inC: 2, tIn: 2, outC: 1, kSize: 1, stride: 1, pad: 0, dilation: 1, tOut: 2);

            // t0: 2·1 + 10·3 = 32 ; t1: 2·2 + 10·4 = 44
            Assert.Equal([32f, 44f], dst);
        }

        [Theory]
        // tIn, k, stride, pad, dilation, outputPadding  → expected length per the PyTorch formula
        [InlineData(2, 3, 2, 0, 1, 0)]
        [InlineData(10, 4, 2, 1, 1, 0)]
        [InlineData(5, 1, 1, 0, 1, 0)]
        [InlineData(7, 3, 2, 1, 1, 1)]   // output_padding 1
        [InlineData(4, 3, 1, 0, 2, 0)]   // dilation 2
        public void OutputLength_MatchesPyTorchFormula(int tIn, int k, int stride, int pad, int dil, int outPad)
        {
            var expected = ((tIn - 1) * stride) - (2 * pad) + (dil * (k - 1)) + outPad + 1;
            Assert.Equal(expected, SnacConv.OutputLength(tIn, k, stride, pad, dil, outPad));
        }

        [Fact]
        public void Bias_AddedToEveryOutputSample()
        {
            // Zero weights → every output sample is exactly its channel bias (incl. positions no input reaches).
            float[] zeroWeight = new float[2 * 2 * 2]; // [inC=2, outC=2, k=2]
            float[] input = [1f, 2f, 3f, 4f, 5f, 6f];  // [inC=2 × tIn=3]
            float[] bias = [0.25f, -0.5f];
            var tOut = SnacConv.OutputLength(3, 2, 1, 0, 1, 0);
            var dst = new float[2 * tOut];

            SnacConv.ConvTranspose1d(input, zeroWeight, bias, dst,
                inC: 2, tIn: 3, outC: 2, kSize: 2, stride: 1, pad: 0, dilation: 1, tOut: tOut);

            for (var t = 0; t < tOut; t++)
            {
                Assert.Equal(0.25f, dst[t], 6);
                Assert.Equal(-0.5f, dst[tOut + t], 6);
            }
        }

        [Theory]
        [InlineData(2, 1, 3, 2, 0, 1)]
        [InlineData(3, 2, 4, 2, 1, 1)]
        [InlineData(1, 4, 5, 3, 2, 1)]
        [InlineData(2, 2, 3, 1, 0, 2)]   // dilation 2
        [InlineData(8, 16, 7, 4, 2, 1)]  // large → crosses the parallel threshold
        public void GatherKernel_MatchesScatterDefinition(int inC, int outC, int kSize, int stride, int pad, int dil)
        {
            const int tIn = 40;
            var tOut = SnacConv.OutputLength(tIn, kSize, stride, pad, dil, outputPadding: 0);
            var input = Random(inC * tIn, seed: 11);
            var weight = Random(inC * outC * kSize, seed: 22);
            var bias = Random(outC, seed: 33);

            var actual = new float[outC * tOut];
            SnacConv.ConvTranspose1d(input, weight, bias, actual, inC, tIn, outC, kSize, stride, pad, dil, tOut);

            var expected = ScatterReference(input, weight, bias, inC, tIn, outC, kSize, stride, pad, dil, tOut);

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], 4);
            }
        }

        [Fact]
        public void Conv1d_Depthwise_IsPerChannelConvolution()
        {
            // groups=channels: each channel convolved by its own kernel, independently. 2 channels, k=2, no pad.
            float[] input = [1f, 2f, 3f, /* ch1 */ 10f, 20f, 30f]; // [inC=2 × tIn=3]
            float[] weight = [1f, 1f, /* ch1 kernel */ 0f, 2f];     // [outC=2 × icPerGroup=1 × k=2]
            var tOut = SnacConv.ConvOutputLength(3, 2, 1, 0, 1);
            var dst = new float[2 * tOut];

            SnacConv.Conv1d(input, weight, ReadOnlySpan<float>.Empty, dst,
                inC: 2, tIn: 3, outC: 2, kSize: 2, stride: 1, pad: 0, dilation: 1, groups: 2, tOut: tOut);

            Assert.Equal(2, tOut);
            // ch0 kernel [1,1]: (1+2),(2+3)=3,5 ; ch1 kernel [0,2]: (0·10+2·20),(0·20+2·30)=40,60
            Assert.Equal([3f, 5f], dst[..2]);
            Assert.Equal([40f, 60f], dst[2..]);
        }

        [Theory]
        [InlineData(4, 4, 7, 1, 3, 1, 1)]    // groups=1 regular, "same" padding
        [InlineData(6, 6, 7, 1, 9, 3, 6)]    // depthwise (groups=channels), dilation 3
        [InlineData(8, 4, 3, 2, 1, 1, 2)]    // grouped (2 groups), strided
        [InlineData(512, 512, 1, 1, 0, 1, 1)] // pointwise 1x1, large → crosses parallel threshold
        public void Conv1d_MatchesNaiveReference(int inC, int outC, int kSize, int stride, int pad, int dil, int groups)
        {
            const int tIn = 30;
            var tOut = SnacConv.ConvOutputLength(tIn, kSize, stride, pad, dil);
            var icPerGroup = inC / groups;
            var input = Random(inC * tIn, seed: 101);
            var weight = Random(outC * icPerGroup * kSize, seed: 202);
            var bias = Random(outC, seed: 303);

            var actual = new float[outC * tOut];
            SnacConv.Conv1d(input, weight, bias, actual, inC, tIn, outC, kSize, stride, pad, dil, groups, tOut);

            var expected = NaiveConv1d(input, weight, bias, inC, tIn, outC, kSize, stride, pad, dil, groups, tOut);
            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], 4);
            }
        }

        private static float[] NaiveConv1d(
            float[] input, float[] weight, float[] bias, int inC, int tIn, int outC, int kSize,
            int stride, int pad, int dilation, int groups, int tOut)
        {
            var icPerGroup = inC / groups;
            var ocPerGroup = outC / groups;
            var dst = new float[outC * tOut];
            for (var oc = 0; oc < outC; oc++)
            {
                var g = oc / ocPerGroup;
                for (var to = 0; to < tOut; to++)
                {
                    var acc = bias[oc];
                    for (var icl = 0; icl < icPerGroup; icl++)
                    {
                        for (var k = 0; k < kSize; k++)
                        {
                            var ti = (to * stride) - pad + (k * dilation);
                            if (ti >= 0 && ti < tIn)
                            {
                                acc += weight[((oc * icPerGroup) + icl) * kSize + k] * input[((g * icPerGroup) + icl) * tIn + ti];
                            }
                        }
                    }
                    dst[(oc * tOut) + to] = acc;
                }
            }
            return dst;
        }

        // Canonical transposed-conv definition: scatter each (input, kernel) pair onto its output position. An
        // independent formulation from the kernel's gather loop, so agreement validates the gather indexing math.
        private static float[] ScatterReference(
            float[] input, float[] weight, float[] bias, int inC, int tIn, int outC, int kSize,
            int stride, int pad, int dilation, int tOut)
        {
            var dst = new float[outC * tOut];
            for (var oc = 0; oc < outC; oc++)
            {
                for (var to = 0; to < tOut; to++)
                {
                    dst[(oc * tOut) + to] = bias[oc];
                }
            }

            for (var ic = 0; ic < inC; ic++)
            {
                for (var ti = 0; ti < tIn; ti++)
                {
                    var x = input[(ic * tIn) + ti];
                    for (var oc = 0; oc < outC; oc++)
                    {
                        for (var k = 0; k < kSize; k++)
                        {
                            var to = (ti * stride) + (k * dilation) - pad;
                            if (to >= 0 && to < tOut)
                            {
                                dst[(oc * tOut) + to] += weight[((ic * outC) + oc) * kSize + k] * x;
                            }
                        }
                    }
                }
            }
            return dst;
        }

        private static float[] Random(int n, uint seed)
        {
            var a = new float[n];
            var state = seed;
            for (var i = 0; i < n; i++)
            {
                state = (state * 1664525u) + 1013904223u;
                a[i] = ((state >> 8) / (float)(1 << 24)) - 0.5f; // [-0.5, 0.5)
            }
            return a;
        }
    }
}
