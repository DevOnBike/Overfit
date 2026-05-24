// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Min-P sampling (<see cref="SamplingStrategy.MinP"/>): a token survives only if its
    /// probability ≥ MinP × P(top); sampling then draws from the survivors. Deterministic
    /// checks on a crafted logit distribution.
    /// </summary>
    public sealed class TokenSamplerMinPTests
    {
        // logits → P(0):P(1) ratio = exp(9.9-10) ≈ 0.905; everything else ~exp(-10) ≈ 4.5e-5.
        private static float[] Logits()
        {
            var l = new float[10];
            l[0] = 10f;
            l[1] = 9.9f;
            return l;   // rest 0
        }

        [Fact]
        public void MinP_Moderate_KeepsTopCluster_ExcludesLongTail()
        {
            var logits = Logits();
            var idx = new int[logits.Length];
            var sc = new float[logits.Length];
            var rng = new Random(7);
            var opts = SamplingOptions.WithMinP(0.5f, temperature: 1f);   // 0.905 ≥ 0.5 → {0,1} survive

            for (var draw = 0; draw < 300; draw++)
            {
                var t = TokenSampler.Sample(logits, in opts, rng, idx, sc);
                Assert.True(t is 0 or 1, $"Min-P sampled a long-tail token {t}.");
            }
        }

        [Fact]
        public void MinP_Strict_KeepsOnlyTop()
        {
            var logits = Logits();
            var idx = new int[logits.Length];
            var sc = new float[logits.Length];
            var rng = new Random(7);
            var opts = SamplingOptions.WithMinP(0.95f, temperature: 1f);  // 0.905 < 0.95 → only {0}

            for (var draw = 0; draw < 50; draw++)
            {
                Assert.Equal(0, TokenSampler.Sample(logits, in opts, rng, idx, sc));
            }
        }

        [Fact]
        public void MinP_DisabledWhenZero_FallsBackToFullSoftmax()
        {
            // MinP strategy with MinP=0 must not crash and must return a valid token.
            var logits = Logits();
            var idx = new int[logits.Length];
            var sc = new float[logits.Length];
            var opts = new SamplingOptions(SamplingStrategy.MinP, temperature: 1f, topK: 0, topP: 1f, seed: 0, minP: 0f);

            var t = TokenSampler.Sample(logits, in opts, new Random(1), idx, sc);
            Assert.InRange(t, 0, logits.Length - 1);
        }
    }
}
