// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// Correctness tests for <see cref="CtcLoss"/>. The anchor is a finite-difference gradient check:
    /// the analytic <c>softmax − posterior</c> gradient must match the numerical derivative of the loss,
    /// which validates the whole forward–backward implementation regardless of derivation. Plus a few
    /// closed-form / monotonicity / edge cases.
    /// </summary>
    public sealed class CtcLossTests
    {
        [Fact]
        public void EmptyTarget_SingleStep_IsNegLogBlankProb()
        {
            // T=1, C=3, blank=2, empty target ⇒ loss = −log softmax(blank).
            const int t = 1, c = 3, blank = 2;
            var logits = new[] { 0.5f, -0.2f, 1.0f };
            var loss = CtcLoss.Forward(logits, t, c, ReadOnlySpan<int>.Empty, blank, default);

            // softmax(blank) for [0.5,-0.2,1.0]:
            var max = 1.0f;
            var z = MathF.Exp(0.5f - max) + MathF.Exp(-0.2f - max) + MathF.Exp(1.0f - max);
            var expected = -(1.0f - max - MathF.Log(z));
            Assert.Equal(expected, loss, 4);
        }

        [Fact]
        public void ImpossibleAlignment_ReturnsPositiveInfinity()
        {
            // target "a a" needs a separating blank ⇒ at least 3 timesteps; give it 2.
            const int t = 2, c = 3, blank = 2;
            var logits = new float[t * c];
            var loss = CtcLoss.Forward(logits, t, c, new[] { 0, 0 }, blank, default);
            Assert.True(float.IsPositiveInfinity(loss));
        }

        [Fact]
        public void Gradient_MatchesFiniteDifference()
        {
            const int t = 5, c = 4, blank = 3;
            int[] target = [0, 1, 1, 2]; // includes a repeat (1,1) to exercise the blank-skip rule
            var rng = new Random(1234);
            var logits = new float[t * c];
            for (var i = 0; i < logits.Length; i++)
            {
                logits[i] = (float)(rng.NextDouble() * 2 - 1);
            }

            var grad = new float[t * c];
            var loss = CtcLoss.Forward(logits, t, c, target, blank, grad);
            Assert.True(float.IsFinite(loss));

            const float eps = 1e-3f;
            var maxErr = 0f;
            for (var i = 0; i < logits.Length; i++)
            {
                var save = logits[i];
                logits[i] = save + eps;
                var lp = CtcLoss.Forward(logits, t, c, target, blank, default);
                logits[i] = save - eps;
                var lm = CtcLoss.Forward(logits, t, c, target, blank, default);
                logits[i] = save;

                var fd = (lp - lm) / (2 * eps);
                maxErr = MathF.Max(maxErr, MathF.Abs(fd - grad[i]));
            }

            Assert.True(maxErr < 5e-3f, $"max |analytic − finite-diff| = {maxErr} (expected < 5e-3)");
        }

        [Fact]
        public void GradientDescent_DrivesLossDown_TowardTarget()
        {
            // A tiny "model" = a free logit table; SGD on the CTC gradient must lower the loss and make
            // the most-probable non-blank emissions spell the target.
            const int t = 6, c = 4, blank = 3;
            int[] target = [0, 1, 2];
            var rng = new Random(7);
            var logits = new float[t * c];
            for (var i = 0; i < logits.Length; i++)
            {
                logits[i] = (float)(rng.NextDouble() * 0.1);
            }

            var grad = new float[t * c];
            var first = CtcLoss.Forward(logits, t, c, target, blank, grad);

            var last = first;
            for (var step = 0; step < 200; step++)
            {
                last = CtcLoss.Forward(logits, t, c, target, blank, grad);
                for (var i = 0; i < logits.Length; i++)
                {
                    logits[i] -= 1.0f * grad[i];
                }
            }

            Assert.True(last < first, $"loss did not drop: {first} -> {last}");
            Assert.True(last < 0.05f, $"loss did not converge: {last}");

            // Greedy collapse (argmax per step, drop blanks + repeats) should recover the target.
            var decoded = GreedyDecode(logits, t, c, blank);
            Assert.Equal(target, decoded);
        }

        private static int[] GreedyDecode(float[] logits, int t, int c, int blank)
        {
            var outp = new List<int>();
            var prev = -1;
            for (var ti = 0; ti < t; ti++)
            {
                var best = 0;
                for (var k = 1; k < c; k++)
                {
                    if (logits[ti * c + k] > logits[ti * c + best])
                    {
                        best = k;
                    }
                }
                if (best != blank && best != prev)
                {
                    outp.Add(best);
                }
                prev = best;
            }
            return outp.ToArray();
        }
    }
}
