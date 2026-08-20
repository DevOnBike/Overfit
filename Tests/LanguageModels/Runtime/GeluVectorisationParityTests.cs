// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// The vectorised GELU must compute the same function as the scalar form it replaced.
    ///
    /// <para><b>The oracle is written out in full here on purpose.</b> It is the exact loop that
    /// <c>CachedFeedForwardBlock.ApplyGeLU</c> shipped before 2026-08-19 —
    /// <c>0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))</c> — and it is spelled out rather than
    /// called, so that the subject and the oracle cannot become the same code by a later edit. A
    /// self-consistency assertion whose two sides share a variable is blind by construction; that is
    /// <c>XC-59</c> and it is the reason this file does not simply invoke the production method twice.</para>
    ///
    /// <para><b>Why a tolerance and not equality.</b> The rewrite is the algebraic identity
    /// <c>0.5 * (1 + tanh(z)) = sigmoid(2z)</c>, so the two agree mathematically, but they evaluate
    /// different floating-point expressions through different library routines. A few ULP of disagreement is
    /// expected and is the same caveat <c>ApplySiLU</c> already carries. The bound below is absolute because
    /// GELU's output crosses zero, where a relative bound is meaningless.</para>
    /// </summary>
    public sealed class GeluVectorisationParityTests
    {
        /// <summary>
        /// Absolute agreement required. The measured maximum over the cases below is roughly 3e-7, so this
        /// leaves better than an order of magnitude of headroom while still being far tighter than any
        /// change of formula could slip through — swapping the tanh approximation for the erf form, for
        /// instance, moves the output by about 1e-3.
        /// </summary>
        private const float Tolerance = 1e-5f;

        private static float ScalarReference(float x)
        {
            const float sqrtTwoOverPi = 0.7978845608028654f;
            const float coeff = 0.044715f;

            var x3 = x * x * x;
            var inner = sqrtTwoOverPi * (x + (coeff * x3));

            return 0.5f * x * (1f + MathF.Tanh(inner));
        }

        [Fact]
        public void TheVectorisedFormMatchesTheScalarOne_AcrossTheRangeAnActivationActuallySees()
        {
            var rng = new Random(20260819);
            var values = new float[4096];
            var expected = new float[values.Length];

            for (var i = 0; i < values.Length; i++)
            {
                values[i] = (float)((rng.NextDouble() * 16.0) - 8.0);
                expected[i] = ScalarReference(values[i]);
            }

            CachedFeedForwardBlock.ApplyGeLU(values);

            var worst = 0f;
            var worstAt = -1;

            for (var i = 0; i < values.Length; i++)
            {
                var difference = MathF.Abs(values[i] - expected[i]);

                if (difference > worst)
                {
                    worst = difference;
                    worstAt = i;
                }
            }

            Assert.True(worst <= Tolerance,
                $"vectorised GELU differs from the scalar form by {worst:E3} at index {worstAt} "
                + $"(tolerance {Tolerance:E3})");
        }

        [Theory]
        // The transition around zero, where the two formulations have the most room to disagree.
        [InlineData(0f)]
        [InlineData(0.5f)]
        [InlineData(-0.5f)]
        [InlineData(1f)]
        [InlineData(-1f)]
        // The tails, where GELU must approach x and 0 respectively. `sigmoid` saturates here and `tanh`
        // saturates there, so this is where an identity that was only ALMOST right would show.
        [InlineData(12f)]
        [InlineData(-12f)]
        [InlineData(40f)]
        [InlineData(-40f)]
        public void TheTwoFormsAgreeAtTheBoundaries(float x)
        {
            var values = new[] { x };
            var expected = ScalarReference(x);

            CachedFeedForwardBlock.ApplyGeLU(values);

            Assert.True(MathF.Abs(values[0] - expected) <= Tolerance,
                $"at x={x}: vectorised {values[0]:R} against scalar {expected:R}");
        }

        [Fact]
        public void AnEmptyBufferIsLeftAlone()
        {
            // The vectorised form rents a scratch buffer sized to the input. Zero is the size at which a
            // pool rental is most likely to be handled specially, so it is checked rather than assumed.
            var values = Array.Empty<float>();

            CachedFeedForwardBlock.ApplyGeLU(values);

            Assert.Empty(values);
        }
    }
}
