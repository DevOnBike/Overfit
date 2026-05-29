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
    /// The trained conv path now supports padding, stride and bias (previously VALID / no-bias only).
    /// Two checks: (1) the graph forward matches the independent <c>ForwardInference</c> NCHW kernel, and
    /// (2) the backward gradients (input, kernel, bias) match finite differences of the loss.
    /// </summary>
    public sealed class ConvLayerPaddingStrideBiasTests
    {
        private const int InC = 2, OutC = 3, Hh = 5, Ww = 5, K = 3, Pad = 1, Stride = 2;

        private static void Fill(Span<float> span, Random rng)
        {
            for (var i = 0; i < span.Length; i++) { span[i] = (float)(rng.NextDouble() * 2 - 1); }
        }

        private static ConvLayer MakeConv(Random rng)
        {
            var conv = new ConvLayer(InC, OutC, Hh, Ww, K, Pad, Stride, useBias: true);
            Fill(conv.Kernels.DataSpan, rng);
            Fill(conv.Bias!.DataSpan, rng);
            conv.Train();
            return conv;
        }

        [Fact]
        public void GraphForward_MatchesInferenceKernel()
        {
            var rng = new Random(11);
            using var conv = MakeConv(rng);
            using var graph = new ComputationGraph(1 << 20);

            var inputData = new float[InC * Hh * Ww];
            Fill(inputData, rng);

            // Graph (training) forward.
            var store = new TensorStorage<float>(inputData.Length, clearMemory: false);
            inputData.AsSpan().CopyTo(store.AsSpan());
            using var input = new AutogradNode(store, new TensorShape(1, InC, Hh, Ww), requiresGrad: false);
            var outNode = conv.Forward(graph, input);

            // Independent inference kernel (Conv2DKernels NCHW + bias).
            var outH = (Hh + 2 * Pad - K) / Stride + 1;
            var outW = (Ww + 2 * Pad - K) / Stride + 1;
            var reference = new float[OutC * outH * outW];
            conv.ForwardInference(inputData, reference);

            Assert.Equal(reference.Length, outNode.DataView.AsReadOnlySpan().Length);
            var graphOut = outNode.DataView.AsReadOnlySpan();
            var maxDiff = 0f;
            for (var i = 0; i < reference.Length; i++)
            {
                maxDiff = MathF.Max(maxDiff, MathF.Abs(graphOut[i] - reference[i]));
            }
            Assert.True(maxDiff < 1e-4f, $"graph forward vs inference kernel differ by {maxDiff}");
        }

        [Fact]
        public void Backward_GradientsMatchFiniteDifference()
        {
            var rng = new Random(13);
            using var conv = MakeConv(rng);
            using var graph = new ComputationGraph(1 << 20);

            var inLen = InC * Hh * Ww;
            var inputData = new float[inLen];
            Fill(inputData, rng);

            // Loss = sum(output) ⇒ dL/dOutput = 1.
            float Loss()
            {
                graph.Reset();
                var store = new TensorStorage<float>(inLen, clearMemory: false);
                inputData.AsSpan().CopyTo(store.AsSpan());
                using var input = new AutogradNode(store, new TensorShape(1, InC, Hh, Ww), requiresGrad: false);
                var outp = conv.Forward(graph, input);
                var s = 0f;
                foreach (var v in outp.DataView.AsReadOnlySpan()) { s += v; }
                return s;
            }

            // Analytic gradients.
            graph.Reset();
            conv.Kernels.GradSpan.Clear();
            conv.Bias!.GradSpan.Clear();
            var store2 = new TensorStorage<float>(inLen, clearMemory: false);
            inputData.AsSpan().CopyTo(store2.AsSpan());
            using var input2 = new AutogradNode(store2, new TensorShape(1, InC, Hh, Ww), requiresGrad: true);
            var outNode = conv.Forward(graph, input2);
            outNode.GradView.AsSpan().Fill(1f);
            graph.BackwardFromGrad(outNode);

            var inGrad = input2.GradView.AsReadOnlySpan().ToArray();
            var kGrad = conv.Kernels.GradSpan.ToArray();
            var bGrad = conv.Bias!.GradSpan.ToArray();

            const float eps = 1e-3f;

            CheckFiniteDiff("input", inputData, inGrad, eps, Loss);
            CheckFiniteDiff("kernel", conv.Kernels.DataSpan, kGrad, eps, Loss);
            CheckFiniteDiff("bias", conv.Bias!.DataSpan, bGrad, eps, Loss);
        }

        private static void CheckFiniteDiff(string name, Span<float> values, float[] analytic, float eps, Func<float> loss)
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

            Assert.True(maxErr < 1e-2f * maxMag + 1e-2f,
                $"{name}: finite-diff vs analytic max error {maxErr} (max |grad| {maxMag}) too high");
        }

        [Fact]
        public void SamePadding_PreservesSpatialDims()
        {
            using var conv = new ConvLayer(inChannels: 1, outChannels: 4, h: 8, w: 8, kSize: 3, padding: 1, stride: 1, useBias: true);
            conv.Eval();
            using var graph = new ComputationGraph(1 << 18);

            var store = new TensorStorage<float>(64, clearMemory: true);
            using var input = new AutogradNode(store, new TensorShape(1, 1, 8, 8), requiresGrad: false);
            var outNode = conv.Forward(graph, input);

            Assert.Equal(4 * 8 * 8, outNode.Shape.Size);   // SAME conv keeps 8×8
        }
    }
}
