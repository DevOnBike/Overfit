// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// <see cref="DepthwiseConv2DLayer"/>: SIMD forward matches an independent naive reference (SAME and
    /// strided), and input/kernel/bias gradients match finite differences.
    /// </summary>
    public sealed class DepthwiseConv2DTests
    {
        private const int N = 2, C = 3, H = 5, W = 5, K = 3;

        private static void Fill(Span<float> s, Random rng)
        {
            for (var i = 0; i < s.Length; i++)
            {
                s[i] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        private static AutogradNode NewNode(float[] data, TensorShape shape, bool grad)
        {
            var store = new TensorStorage<float>(data.Length, clearMemory: false);
            data.AsSpan().CopyTo(store.AsSpan());
            return new AutogradNode(store, shape, requiresGrad: grad);
        }

        private static float[] NaiveDepthwise(
            float[] inp, float[] kernel, float[] bias, int p, int s)
        {
            var outH = (H + 2 * p - K) / s + 1;
            var outW = (W + 2 * p - K) / s + 1;
            var kk = K * K;
            var outp = new float[N * C * outH * outW];
            for (var n = 0; n < N; n++)
            {
                for (var c = 0; c < C; c++)
                {
                    for (var oy = 0; oy < outH; oy++)
                    {
                        for (var ox = 0; ox < outW; ox++)
                        {
                            var acc = bias is null ? 0f : bias[c];
                            for (var ky = 0; ky < K; ky++)
                            {
                                for (var kx = 0; kx < K; kx++)
                                {
                                    var iy = oy * s - p + ky;
                                    var ix = ox * s - p + kx;
                                    if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                    {
                                        acc += inp[((n * C + c) * H + iy) * W + ix] * kernel[c * kk + ky * K + kx];
                                    }
                                }
                            }
                            outp[((n * C + c) * outH + oy) * outW + ox] = acc;
                        }
                    }
                }
            }
            return outp;
        }

        [Theory]
        [InlineData(1, 1)]   // SAME (padding 1, stride 1) — SIMD path
        [InlineData(0, 1)]   // VALID
        [InlineData(1, 2)]   // strided — scalar-fallback path
        public void Forward_MatchesNaiveReference(int padding, int stride)
        {
            var rng = new Random(padding * 31 + stride);
            using var dw = new DepthwiseConv2DLayer(C, H, W, K, padding, stride, useBias: true);
            Fill(dw.Kernels.DataSpan, rng);
            Fill(dw.Bias!.DataSpan, rng);
            using var graph = new ComputationGraph(1 << 18);

            var data = new float[N * C * H * W];
            Fill(data, rng);
            using var input = NewNode(data, new TensorShape(N, C, H, W), false);
            var outNode = dw.Forward(graph, input);

            var reference = NaiveDepthwise(data, dw.Kernels.DataSpan.ToArray(), dw.Bias!.DataSpan.ToArray(), padding, stride);
            Assert.Equal(reference.Length, outNode.DataView.AsReadOnlySpan().Length);
            var got = outNode.DataView.AsReadOnlySpan();
            var maxDiff = 0f;
            for (var i = 0; i < reference.Length; i++)
            {
                maxDiff = MathF.Max(maxDiff, MathF.Abs(got[i] - reference[i]));
            }
            Assert.True(maxDiff < 1e-4f, $"depthwise forward vs reference differ by {maxDiff} (p={padding}, s={stride})");
        }

        [Fact]
        public void Backward_GradientsMatchFiniteDifference()
        {
            const int pad = 1, stride = 1;
            var rng = new Random(17);
            using var dw = new DepthwiseConv2DLayer(C, H, W, K, pad, stride, useBias: true);
            Fill(dw.Kernels.DataSpan, rng);
            Fill(dw.Bias!.DataSpan, rng);
            using var graph = new ComputationGraph(1 << 18);

            var data = new float[N * C * H * W];
            Fill(data, rng);

            float Loss()
            {
                graph.Reset();
                using var input = NewNode(data, new TensorShape(N, C, H, W), false);
                var outp = dw.Forward(graph, input);
                var s = 0f;
                foreach (var v in outp.DataView.AsReadOnlySpan())
                {
                    s += v;
                }
                return s;
            }

            graph.Reset();
            dw.Kernels.GradSpan.Clear();
            dw.Bias!.GradSpan.Clear();
            using var input2 = NewNode(data, new TensorShape(N, C, H, W), true);
            var outNode = dw.Forward(graph, input2);
            outNode.GradView.AsSpan().Fill(1f);
            graph.BackwardFromGrad(outNode);

            var inGrad = input2.GradView.AsReadOnlySpan().ToArray();
            var kGrad = dw.Kernels.GradSpan.ToArray();
            var bGrad = dw.Bias!.GradSpan.ToArray();

            const float eps = 1e-3f;
            CheckFd("input", data, inGrad, eps, Loss);
            CheckFd("kernel", dw.Kernels.DataSpan, kGrad, eps, Loss);
            CheckFd("bias", dw.Bias!.DataSpan, bGrad, eps, Loss);
        }

        private static void CheckFd(string name, Span<float> values, float[] analytic, float eps, Func<float> loss)
        {
            var maxErr = 0f;
            var maxMag = 0f;
            for (var i = 0; i < values.Length; i++)
            {
                var saved = values[i];
                values[i] = saved + eps;
                var lp = loss();
                values[i] = saved - eps;
                var lm = loss();
                values[i] = saved;
                var fd = (lp - lm) / (2 * eps);
                maxErr = MathF.Max(maxErr, MathF.Abs(fd - analytic[i]));
                maxMag = MathF.Max(maxMag, MathF.Abs(analytic[i]));
            }
            Assert.True(maxErr < 1e-2f * maxMag + 1e-2f, $"{name}: FD vs analytic max error {maxErr} (max|grad| {maxMag})");
        }
    }
}
