// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Proves <see cref="SpeculativeSampler.AcceptOrResample"/> is sampling-correct: for ANY draft token,
    /// the accept-or-resample output is distributed exactly as the target distribution <c>p</c>. Checked
    /// statistically (many trials, fixed seed) — the empirical distribution must match <c>p</c> regardless
    /// of which token was drafted, which is the guarantee that makes speculative decoding distribution-
    /// preserving.
    /// </summary>
    public sealed class SpeculativeSamplerTests
    {
        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        public void AcceptOrResample_OutputIsDistributedAsTarget_ForAnyDraft(int draft)
        {
            float[] p = [0.10f, 0.55f, 0.25f, 0.10f];
            const int trials = 400_000;
            var rng = new Random(12345);
            var work = new float[p.Length];
            var counts = new int[p.Length];

            for (var t = 0; t < trials; t++)
            {
                counts[SpeculativeSampler.AcceptOrResample(p, draft, rng, work)]++;
            }

            // Empirical distribution must match p within Monte-Carlo noise — independent of the draft.
            for (var i = 0; i < p.Length; i++)
            {
                var empirical = (float)counts[i] / trials;
                Assert.True(MathF.Abs(empirical - p[i]) < 0.01f,
                    $"token {i}: empirical {empirical:F4} vs target {p[i]:F4} (draft={draft})");
            }
        }

        [Fact]
        public void AcceptOrResample_PointMassTarget_AlwaysReturnsThatToken()
        {
            // p = e_2 (greedy / T→0 limit). Whatever the draft, the output must be token 2.
            float[] p = [0f, 0f, 1f, 0f];
            var rng = new Random(7);
            var work = new float[4];
            for (var draft = 0; draft < 4; draft++)
            {
                Assert.Equal(2, SpeculativeSampler.AcceptOrResample(p, draft, rng, work));
            }
        }

        [Fact]
        public void Sample_DrawsProportionally()
        {
            float[] p = [0.2f, 0.0f, 0.8f];
            const int trials = 200_000;
            var rng = new Random(99);
            var counts = new int[p.Length];
            for (var t = 0; t < trials; t++)
            {
                counts[SpeculativeSampler.Sample(p, rng)]++;
            }

            Assert.True(MathF.Abs((float)counts[0] / trials - 0.2f) < 0.01f);
            Assert.Equal(0, counts[1]);   // zero-probability token never drawn
            Assert.True(MathF.Abs((float)counts[2] / trials - 0.8f) < 0.01f);
        }
    }
}
