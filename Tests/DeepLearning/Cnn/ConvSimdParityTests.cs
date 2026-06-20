// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.DeepLearning.Cnn
{
    /// <summary>
    /// Validates the unit-stride SIMD conv path (the output-row scalar·vector FMA accumulation) against an
    /// independent naive scalar reference. The SIMD reduction order differs from per-pixel scalar, so they agree
    /// within float tolerance, not bit-identically — this pins that the reorder stays numerically faithful.
    /// </summary>
    public sealed class ConvSimdParityTests
    {
        [Fact]
        public void Padded3x3Stride1_SimdMatchesReference()
        {
            const int inC = 8, outC = 6, h = 12, w = 14, k = 3, pad = 1, stride = 1;
            var input = Deterministic(inC * h * w, 11);
            var kernels = Deterministic(outC * inC * k * k, 22);

            var outH = (h + 2 * pad - k) / stride + 1;
            var outW = (w + 2 * pad - k) / stride + 1;
            var output = new float[outC * outH * outW];

            // Goes through ForwardNchw (padding != 0) → unit-stride SIMD worker.
            Conv2DKernels.ForwardNchw(input, kernels, output, 1, inC, outC, h, w, k, pad, stride);

            var reference = ReferenceConv(input, kernels, inC, outC, h, w, k, pad, stride);
            AssertClose(reference, output, 1e-3f);
        }

        [Fact]
        public void Padded7x7Stride2_GatherSimdMatchesReference()
        {
            // ResNet stem shape: 7x7, stride 2, pad 3 → exercises the AVX2 gather strided worker.
            const int inC = 3, outC = 8, h = 32, w = 30, k = 7, pad = 3, stride = 2;
            var input = Deterministic(inC * h * w, 55);
            var kernels = Deterministic(outC * inC * k * k, 66);

            var outH = (h + 2 * pad - k) / stride + 1;
            var outW = (w + 2 * pad - k) / stride + 1;
            var output = new float[outC * outH * outW];

            Conv2DKernels.ForwardNchw(input, kernels, output, 1, inC, outC, h, w, k, pad, stride);

            var reference = ReferenceConv(input, kernels, inC, outC, h, w, k, pad, stride);
            AssertClose(reference, output, 1e-3f);
        }

        [Fact]
        public void OneByOneStride2_GatherSimdMatchesReference()
        {
            // ResNet downsample shape: 1x1, stride 2, pad 0.
            const int inC = 16, outC = 12, h = 14, w = 14, k = 1, pad = 0, stride = 2;
            var input = Deterministic(inC * h * w, 77);
            var kernels = Deterministic(outC * inC * k * k, 88);

            var outH = (h + 2 * pad - k) / stride + 1;
            var outW = (w + 2 * pad - k) / stride + 1;
            var output = new float[outC * outH * outW];

            Conv2DKernels.ForwardNchw(input, kernels, output, 1, inC, outC, h, w, k, pad, stride);

            var reference = ReferenceConv(input, kernels, inC, outC, h, w, k, pad, stride);
            AssertClose(reference, output, 1e-3f);
        }

        [Fact]
        public void OneByOneStride1_SimdMatchesReference()
        {
            // 1x1 conv (pad 0, stride 1) → ForwardValidNchw → rerouted to the unit-stride SIMD worker.
            const int inC = 16, outC = 12, h = 7, w = 7, k = 1, pad = 0, stride = 1;
            var input = Deterministic(inC * h * w, 33);
            var kernels = Deterministic(outC * inC * k * k, 44);

            var output = new float[outC * h * w];
            Conv2DKernels.ForwardValidNchw(input, kernels, output, inC, outC, h, w, k);

            var reference = ReferenceConv(input, kernels, inC, outC, h, w, k, pad, stride);
            AssertClose(reference, output, 1e-3f);
        }

        // Naive, obviously-correct scalar reference: output[oc,oy,ox] = Σ in[ic, oy·s-pad+ky, ox·s-pad+kx]·k[...].
        private static float[] ReferenceConv(
            float[] input, float[] kernels, int inC, int outC, int h, int w, int k, int pad, int stride)
        {
            var outH = (h + 2 * pad - k) / stride + 1;
            var outW = (w + 2 * pad - k) / stride + 1;
            var output = new float[outC * outH * outW];

            for (var oc = 0; oc < outC; oc++)
            {
                for (var oy = 0; oy < outH; oy++)
                {
                    for (var ox = 0; ox < outW; ox++)
                    {
                        double sum = 0;
                        for (var ic = 0; ic < inC; ic++)
                        {
                            for (var ky = 0; ky < k; ky++)
                            {
                                var iy = oy * stride - pad + ky;
                                if (iy < 0 || iy >= h)
                                {
                                    continue;
                                }
                                for (var kx = 0; kx < k; kx++)
                                {
                                    var ix = ox * stride - pad + kx;
                                    if (ix < 0 || ix >= w)
                                    {
                                        continue;
                                    }
                                    sum += (double)input[ic * h * w + iy * w + ix]
                                         * kernels[((oc * inC + ic) * k + ky) * k + kx];
                                }
                            }
                        }
                        output[(oc * outH + oy) * outW + ox] = (float)sum;
                    }
                }
            }

            return output;
        }

        private static void AssertClose(float[] expected, float[] actual, float tol)
        {
            Assert.Equal(expected.Length, actual.Length);
            var maxDiff = 0f;
            for (var i = 0; i < expected.Length; i++)
            {
                maxDiff = MathF.Max(maxDiff, MathF.Abs(expected[i] - actual[i]));
            }
            Assert.True(maxDiff < tol, $"max abs diff {maxDiff} exceeds tolerance {tol}");
        }

        private static float[] Deterministic(int n, int seed)
        {
            var a = new float[n];
            var state = (uint)seed | 1u;
            for (var i = 0; i < n; i++)
            {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                a[i] = (state / (float)uint.MaxValue) * 2f - 1f;
            }
            return a;
        }
    }
}
