// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>
    /// Hand-computed correctness checks for the plain-span Whisper inference kernels (GELU, LayerNorm,
    /// Linear, Conv1d with padding, multi-head attention). Real-model parity is S5; this nails the kernels.
    /// </summary>
    public sealed class WhisperKernelsTests
    {
        [Fact]
        public void Gelu_KnownValues()
        {
            Span<float> x = stackalloc float[] { 0f, 1f, -10f, 10f };
            WhisperKernels.GeluInPlace(x);
            Assert.True(Math.Abs(x[0]) < 1e-6);          // gelu(0) = 0
            Assert.True(Math.Abs(x[1] - 0.8412f) < 1e-3);// tanh-approx gelu(1)
            Assert.True(Math.Abs(x[2]) < 1e-3);          // gelu(−10) ≈ 0
            Assert.True(Math.Abs(x[3] - 10f) < 1e-3);    // gelu(10) ≈ 10
        }

        [Fact]
        public void Linear_WithBias()
        {
            ReadOnlySpan<float> x = stackalloc float[] { 1f, 2f, 3f };          // 1 row, inDim 3
            ReadOnlySpan<float> w = stackalloc float[] { 1, 0, 0, 0, 1, 0 };     // [2 × 3]
            ReadOnlySpan<float> b = stackalloc float[] { 10f, 20f };
            Span<float> dst = stackalloc float[2];
            WhisperKernels.Linear(x, w, b, dst, rows: 1, inDim: 3, outDim: 2);
            Assert.Equal(11f, dst[0], 4);
            Assert.Equal(22f, dst[1], 4);
        }

        [Fact]
        public void LayerNorm_NormalizesRow()
        {
            ReadOnlySpan<float> x = stackalloc float[] { 1f, 2f, 3f };
            ReadOnlySpan<float> g = stackalloc float[] { 1f, 1f, 1f };
            ReadOnlySpan<float> beta = stackalloc float[] { 0f, 0f, 0f };
            Span<float> dst = stackalloc float[3];
            WhisperKernels.LayerNorm(x, g, beta, dst, rows: 1, dim: 3, eps: 0f);
            Assert.True(Math.Abs(dst[0] - (-1.22474f)) < 1e-3);
            Assert.True(Math.Abs(dst[1]) < 1e-5);
            Assert.True(Math.Abs(dst[2] - 1.22474f) < 1e-3);
        }

        [Fact]
        public void Conv1d_WithPadding()
        {
            ReadOnlySpan<float> input = stackalloc float[] { 1f, 2f, 3f };   // [1 ch × 3]
            ReadOnlySpan<float> weight = stackalloc float[] { 1f, 1f, 1f };  // [1 × 1 × 3]
            Span<float> dst = stackalloc float[3];
            WhisperKernels.Conv1d(input, weight, ReadOnlySpan<float>.Empty, dst,
                inC: 1, tIn: 3, outC: 1, kSize: 3, stride: 1, pad: 1, tOut: 3);
            // moving sum with zero-pad edges: [1+2, 1+2+3, 2+3] = [3, 6, 5]
            Assert.Equal(new[] { 3f, 6f, 5f }, dst.ToArray());
        }

        [Fact]
        public void MultiHeadAttention_ZeroQuery_AveragesValues()
        {
            // wq = 0 → all scores equal → uniform softmax → each output row = mean(V).
            const int d = 2, t = 2, heads = 1;
            ReadOnlySpan<float> x = stackalloc float[] { 2f, 0f, 0f, 4f }; // V rows [2,0],[0,4] (identity proj)
            ReadOnlySpan<float> zero = stackalloc float[d * d];            // wq = 0
            ReadOnlySpan<float> id = stackalloc float[] { 1, 0, 0, 1 };    // identity [2×2]
            Span<float> dst = stackalloc float[t * d];

            WhisperKernels.MultiHeadAttention(
                x, t, x, t, d, heads,
                zero, ReadOnlySpan<float>.Empty,   // Q (zero weight, no bias)
                id,                                  // K
                id, ReadOnlySpan<float>.Empty,      // V
                id, ReadOnlySpan<float>.Empty,      // out
                dst, causal: false);

            // mean of V rows = [1, 2] for both query tokens.
            Assert.True(Math.Abs(dst[0] - 1f) < 1e-4 && Math.Abs(dst[1] - 2f) < 1e-4);
            Assert.True(Math.Abs(dst[2] - 1f) < 1e-4 && Math.Abs(dst[3] - 2f) < 1e-4);
        }
    }
}
