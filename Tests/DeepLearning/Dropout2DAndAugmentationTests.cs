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
    /// <see cref="Dropout2DLayer"/> drops whole channels (channel-constant mask) + passes the gradient
    /// through that mask, and is identity at inference; <see cref="ImageAugmentation"/> shift/noise behave.
    /// </summary>
    public sealed class Dropout2DAndAugmentationTests
    {
        private const int N = 2, C = 8, H = 3, W = 3, Hw = H * W;

        private static AutogradNode NewNode(float fill, bool grad)
        {
            var store = new TensorStorage<float>(N * C * Hw, clearMemory: false);
            store.AsSpan().Fill(fill);
            return new AutogradNode(store, new TensorShape(N, C, H, W), requiresGrad: grad);
        }

        [Fact]
        public void Dropout2D_MasksWholeChannels_AndPropagatesGradientThroughMask()
        {
            using var graph = new ComputationGraph(1 << 18);
            using var input = NewNode(2f, grad: true);
            var outNode = graph.Dropout2D(input, probability: 0.5f, isTraining: true);
            var o = outNode.DataView.AsReadOnlySpan();

            // Each channel block is uniform: all 0 (dropped) or all in·scale (kept).
            for (var n = 0; n < N; n++)
            {
                for (var c = 0; c < C; c++)
                {
                    var first = o[(n * C + c) * Hw];
                    Assert.True(first == 0f || MathF.Abs(first - 2f * 2f) < 1e-5f, $"channel value {first} unexpected"); // scale = 1/(1-0.5)=2 → 2·2=4
                    for (var i = 1; i < Hw; i++)
                    {
                        Assert.Equal(first, o[(n * C + c) * Hw + i], 5);
                    }
                }
            }

            // Gradient = mask: kept channel → scale (2), dropped → 0.
            outNode.GradView.AsSpan().Fill(1f);
            graph.BackwardFromGrad(outNode);
            var g = input.GradView.AsReadOnlySpan();
            for (var n = 0; n < N; n++)
            {
                for (var c = 0; c < C; c++)
                {
                    var kept = o[(n * C + c) * Hw] != 0f;
                    var expected = kept ? 2f : 0f;
                    Assert.Equal(expected, g[(n * C + c) * Hw], 5);
                }
            }
        }

        [Fact]
        public void Dropout2D_IsIdentity_AtInference()
        {
            using var graph = new ComputationGraph(1 << 18);
            using var input = NewNode(1.5f, grad: false);
            var outNode = graph.Dropout2D(input, probability: 0.5f, isTraining: false);
            foreach (var v in outNode.DataView.AsReadOnlySpan()) { Assert.Equal(1.5f, v, 5); }
        }

        [Fact]
        public void RandomShift_ZeroShift_IsIdentity_AndNoise_PerturbsBounded()
        {
            var rng = new Random(9);
            var src = new float[1 * 4 * 4];
            for (var i = 0; i < src.Length; i++) { src[i] = i; }
            var dst = new float[src.Length];

            ImageAugmentation.RandomShift(src, dst, channels: 1, height: 4, width: 4, maxShift: 0, rng);
            Assert.Equal(src, dst);

            var noisy = (float[])src.Clone();
            ImageAugmentation.AddGaussianNoise(noisy, sigma: 0f, rng);
            Assert.Equal(src, noisy);                       // sigma 0 → unchanged

            ImageAugmentation.AddGaussianNoise(noisy, sigma: 1f, rng);
            var changed = 0;
            for (var i = 0; i < src.Length; i++)
            {
                Assert.True(float.IsFinite(noisy[i]));
                if (MathF.Abs(noisy[i] - src[i]) > 1e-6f) { changed++; }
            }
            Assert.True(changed > 0, "Gaussian noise changed nothing");
        }
    }
}
