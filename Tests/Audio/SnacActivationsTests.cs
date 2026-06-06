// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Snac;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>SNAC's Snake1d nonlinearity, checked exactly against its definition
    /// <c>x + (α + 1e-9)⁻¹·sin(αx)²</c>: the α=1 closed form at known points, per-channel α applied independently,
    /// and the fixed points where <c>sin(αx)=0</c>. Model-free.</summary>
    public sealed class SnacActivationsTests
    {
        [Fact]
        public void Snake_AlphaOne_MatchesClosedForm()
        {
            // α=1 → f(x) = x + sin(x)^2 (the 1e-9 guard is negligible at α=1).
            float[] x = [0f, 0.5f, 1.0f, -0.7f, 2.0f];
            float[] alpha = [1f];

            SnacActivations.Snake1dInPlace(x, alpha, channels: 1, time: x.Length);

            float[] input = [0f, 0.5f, 1.0f, -0.7f, 2.0f];
            for (var i = 0; i < x.Length; i++)
            {
                var s = MathF.Sin(input[i]);
                Assert.Equal(input[i] + (s * s), x[i], 5);
            }
        }

        [Fact]
        public void Snake_FixedPoints_WhereSineIsZero()
        {
            // sin(α·x)=0 → f(x)=x. With α=1 that's x ∈ {0, π, -π, 2π}.
            float[] x = [0f, MathF.PI, -MathF.PI, 2f * MathF.PI];
            var expected = (float[])x.Clone();
            float[] alpha = [1f];

            SnacActivations.Snake1dInPlace(x, alpha, channels: 1, time: x.Length);

            for (var i = 0; i < x.Length; i++)
            {
                Assert.Equal(expected[i], x[i], 4);
            }
        }

        [Fact]
        public void Snake_PerChannelAlpha_AppliedIndependently()
        {
            // 2 channels × 2 time, distinct α per channel.
            float[] x = [0.3f, 0.6f, /* ch1 */ 0.3f, 0.6f];
            float[] alpha = [1.0f, 2.5f];

            SnacActivations.Snake1dInPlace(x, alpha, channels: 2, time: 2);

            Assert.Equal(Snake(0.3f, 1.0f), x[0], 5);
            Assert.Equal(Snake(0.6f, 1.0f), x[1], 5);
            Assert.Equal(Snake(0.3f, 2.5f), x[2], 5);
            Assert.Equal(Snake(0.6f, 2.5f), x[3], 5);
        }

        private static float Snake(float v, float a)
        {
            var s = MathF.Sin(a * v);
            return v + (1f / (a + 1e-9f) * s * s);
        }
    }
}
