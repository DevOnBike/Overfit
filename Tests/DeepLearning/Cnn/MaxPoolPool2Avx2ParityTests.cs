// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Kernels;

namespace DevOnBike.Overfit.Tests.DeepLearning.Cnn
{
    /// <summary>
    /// Bit-identity gate for the AVX2 pool=2 MaxPool path: values AND argmax indices must match the
    /// scalar pool=2 kernel (the path AVX2 replaces; NOTE: the Generic kernel has a DIFFERENT, first-in-scan-order tie rule — pre-existing divergence, ties only) exactly (indices drive the backward scatter — any tie-rule drift corrupts
    /// gradients silently). Inputs are quantised to one decimal so 2×2 windows are tie-rich.
    /// </summary>
    public sealed class MaxPoolPool2Avx2ParityTests
    {
        [Theory]
        [InlineData(8, 26, 26, 0)]        // the MNIST beast shape
        [InlineData(8, 26, 26, 6272)]     // non-zero batch offset
        [InlineData(3, 32, 32, 0)]        // CIFAR-ish
        [InlineData(1, 16, 64, 123)]      // wide row, odd offset
        [InlineData(2, 14, 14, 0)]        // outW = 7 < 8 → pure scalar tail
        public void Avx2Pool2_MatchesGeneric_ValuesAndIndices(int c, int h, int w, int batchOffset)
        {
            var rnd = new Random(7 + c + w);
            var input = new float[c * h * w];
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = MathF.Round((float)(rnd.NextDouble() * 4 - 2), 1);   // tie-rich
            }

            int outH = h / 2, outW = w / 2;
            var outA = new float[c * outH * outW];
            var idxA = new float[c * outH * outW];
            var outB = new float[c * outH * outW];
            var idxB = new float[c * outH * outW];

            PoolingKernels.MaxPool2DForwardWithIndicesPool2Avx2(
                input, outA, idxA, c, h, w, outH, outW, batchOffset);
            PoolingKernels.MaxPool2DForwardWithIndicesPool2Scalar(
                input, outB, idxB, c, h, w, outH, outW, batchOffset);

            for (var i = 0; i < outA.Length; i++)
            {
                if (outA[i] != outB[i] || idxA[i] != idxB[i])
                {
                    int ow = i % outW, oh = (i / outW) % outH, ch = i / (outW * outH);
                    int r0 = ch * h * w + oh * 2 * w, r1 = r0 + w, col = ow * 2;
                    throw new Xunit.Sdk.XunitException(
                        $"i={i} ch={ch} oh={oh} ow={ow} | val avx={outA[i]} gen={outB[i]} | idx avx={idxA[i]} gen={idxB[i]} | " +
                        $"window r0:[{input[r0 + col]},{input[r0 + col + 1]}] r1:[{input[r1 + col]},{input[r1 + col + 1]}] (r0Start={r0} r1Start={r1})");
                }
            }
        }
    }
}
