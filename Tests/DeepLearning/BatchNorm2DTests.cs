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
    /// <see cref="BatchNorm2D"/>: training-mode forward centers each channel (mean≈0, var≈1 with γ=1,β=0),
    /// and the input/γ/β gradients match finite differences of the loss.
    /// </summary>
    public sealed class BatchNorm2DTests
    {
        private const int N = 4, C = 3, H = 2, W = 2;
        private const int Hw = H * W;
        private const int Total = N * C * Hw;

        private static void Fill(Span<float> s, Random rng)
        {
            for (var i = 0; i < s.Length; i++) { s[i] = (float)(rng.NextDouble() * 4 - 2); }
        }

        private static AutogradNode NewNode(float[] data, TensorShape shape, bool grad)
        {
            var storage = new TensorStorage<float>(data.Length, clearMemory: false);
            data.AsSpan().CopyTo(storage.AsSpan());
            return new AutogradNode(storage, shape, requiresGrad: grad);
        }

        [Fact]
        public void TrainingForward_CentersEachChannel()
        {
            var rng = new Random(3);
            using var bn = new BatchNorm2D(C);   // γ=1, β=0 by default
            bn.Train();
            using var graph = new ComputationGraph(1 << 18);

            var data = new float[Total];
            Fill(data, rng);
            using var input = NewNode(data, new TensorShape(N, C, H, W), false);
            var outNode = bn.Forward(graph, input);
            var o = outNode.DataView.AsReadOnlySpan();

            for (var c = 0; c < C; c++)
            {
                double sum = 0, sumSq = 0;
                for (var n = 0; n < N; n++)
                {
                    for (var i = 0; i < Hw; i++)
                    {
                        var v = o[(n * C + c) * Hw + i];
                        sum += v; sumSq += (double)v * v;
                    }
                }
                var m = N * Hw;
                var mean = sum / m;
                var var = sumSq / m - mean * mean;
                Assert.True(Math.Abs(mean) < 1e-3, $"channel {c} mean {mean} not ~0");
                Assert.True(Math.Abs(var - 1.0) < 1e-2, $"channel {c} var {var} not ~1");
            }
        }

        [Fact]
        public void Backward_GradientsMatchFiniteDifference()
        {
            var rng = new Random(5);
            using var bn = new BatchNorm2D(C);
            Fill(bn.Gamma.DataSpan, rng);
            Fill(bn.Beta.DataSpan, rng);
            bn.Train();
            using var graph = new ComputationGraph(1 << 18);

            var data = new float[Total];
            Fill(data, rng);

            float Loss()
            {
                graph.Reset();
                using var input = NewNode(data, new TensorShape(N, C, H, W), false);
                var outp = bn.Forward(graph, input);
                var s = 0f;
                foreach (var v in outp.DataView.AsReadOnlySpan()) { s += v; }
                return s;
            }

            graph.Reset();
            bn.Gamma.GradSpan.Clear();
            bn.Beta.GradSpan.Clear();
            using var input2 = NewNode(data, new TensorShape(N, C, H, W), true);
            var outNode = bn.Forward(graph, input2);
            outNode.GradView.AsSpan().Fill(1f);
            graph.BackwardFromGrad(outNode);

            var inGrad = input2.GradView.AsReadOnlySpan().ToArray();
            var gGrad = bn.Gamma.GradSpan.ToArray();
            var bGrad = bn.Beta.GradSpan.ToArray();

            const float eps = 1e-3f;
            CheckFd("input", data, inGrad, eps, Loss);
            CheckFd("gamma", bn.Gamma.DataSpan, gGrad, eps, Loss);
            CheckFd("beta", bn.Beta.DataSpan, bGrad, eps, Loss);
        }

        private static void CheckFd(string name, Span<float> values, float[] analytic, float eps, Func<float> loss)
        {
            var maxErr = 0f;
            var maxMag = 0f;
            for (var i = 0; i < values.Length; i++)
            {
                var saved = values[i];
                values[i] = saved + eps; var lp = loss();
                values[i] = saved - eps; var lm = loss();
                values[i] = saved;
                var fd = (lp - lm) / (2 * eps);
                maxErr = MathF.Max(maxErr, MathF.Abs(fd - analytic[i]));
                maxMag = MathF.Max(maxMag, MathF.Abs(analytic[i]));
            }
            Assert.True(maxErr < 1e-2f * maxMag + 1e-2f, $"{name}: FD vs analytic max error {maxErr} (max|grad| {maxMag})");
        }
    }
}
