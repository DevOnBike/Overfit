// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Sampling;

namespace DevOnBike.Overfit.Tests.LanguageModels.Sampling
{
    /// <summary>
    /// Correctness pins for the sampler suite (top-nσ, typical-p, XTC, DRY, Mirostat) — each on a hand-worked
    /// case with a known outcome, so a later perf/refactor pass has a fixed baseline to A/B against.
    /// </summary>
    public sealed class SamplerSuiteTests
    {
        private static int FiniteCount(ReadOnlySpan<float> logits)
        {
            var c = 0;
            foreach (var v in logits)
            {
                if (!float.IsNegativeInfinity(v))
                {
                    c++;
                }
            }
            return c;
        }

        [Fact]
        public void TopNSigma_KeepsOnlyTheOutlierPeak()
        {
            // mean 2.5, σ ≈ 4.33 → threshold = 10 − 1·σ ≈ 5.67; only the 10 survives.
            var logits = new[] { 10f, 0f, 0f, 0f };
            new SamplingPipeline.TopNSigma(1f).Apply(logits);

            Assert.False(float.IsNegativeInfinity(logits[0]));
            Assert.True(float.IsNegativeInfinity(logits[1]));
            Assert.Equal(1, FiniteCount(logits));
        }

        [Fact]
        public void TypicalP_OnUniform_KeepsCumulativeMassP()
        {
            // Uniform: every surprise equals the entropy → deviation 0 → keep tokens until cumulative ≥ p.
            var logits = new float[10]; // all equal → uniform after softmax (0.1 each)
            new SamplingPipeline.TypicalP(0.5f).Apply(logits);

            Assert.Equal(5, FiniteCount(logits)); // 5 × 0.1 = 0.5
        }

        [Fact]
        public void Xtc_MasksTopChoicesAboveThreshold_KeepsLeastProbable()
        {
            // softmax([2,1,-10,-10]) ≈ [0.73, 0.27, ~0, ~0]; two clear 0.1 → mask the 0.73, keep the 0.27.
            var logits = new[] { 2f, 1f, -10f, -10f };
            new SamplingPipeline.Xtc(threshold: 0.1f, probability: 1f, seed: 1).Apply(logits);

            Assert.True(float.IsNegativeInfinity(logits[0]));   // top choice excluded
            Assert.False(float.IsNegativeInfinity(logits[1]));  // least-probable above threshold kept
            Assert.False(float.IsNegativeInfinity(logits[2]));  // below threshold untouched
        }

        [Fact]
        public void Dry_PenalisesTheTokenThatWouldExtendTheRepeat()
        {
            // history a,b,c,a,b  → picking c would repeat "a,b,c" (match length 2 ≥ allowedLength 2).
            var logits = new float[4]; // tokens 0..3
            var history = new[] { 1, 2, 3, 1, 2 };
            new SamplingPipeline.Dry(multiplier: 1f, @base: 2f, allowedLength: 2).Process(logits, history);

            Assert.Equal(-1f, logits[3], 5); // multiplier·base^0 = 1
            Assert.Equal(0f, logits[0]);
            Assert.Equal(0f, logits[1]);
            Assert.Equal(0f, logits[2]);
        }

        [Fact]
        public void MirostatV2_RaisesMuOnLowSurprise_AndPicksThePeak()
        {
            var m = new MirostatSampler(tau: 5f, eta: 0.1f, version: MirostatSampler.Version.V2, seed: 1);
            var before = m.Mu;

            var logits = new[] { 20f, 0f, 0f, 0f }; // near one-hot → surprise ≈ 0 ≪ τ
            var token = m.Sample(logits);

            Assert.Equal(0, token);
            Assert.True(m.Mu > before); // μ ← μ − η(observed − τ), observed ≈ 0 < τ ⇒ μ rises
        }

        [Fact]
        public void MirostatV2_StaysBoundedOverManySteps()
        {
            var m = new MirostatSampler(tau: 4f, eta: 0.1f, version: MirostatSampler.Version.V2, seed: 7);
            var rng = new Random(3);
            for (var step = 0; step < 200; step++)
            {
                var logits = new float[64];
                for (var i = 0; i < logits.Length; i++)
                {
                    logits[i] = (float)(rng.NextDouble() * 6.0);
                }
                var t = m.Sample(logits);
                Assert.InRange(t, 0, logits.Length - 1);
            }
            Assert.True(m.Mu is > 0.0 and < 100.0, $"μ drifted out of range: {m.Mu}");
        }

        [Fact]
        public void MirostatV1_ProducesValidTokens()
        {
            var m = new MirostatSampler(tau: 5f, eta: 0.1f, version: MirostatSampler.Version.V1, seed: 2);
            var rng = new Random(5);
            for (var step = 0; step < 50; step++)
            {
                var logits = new float[128];
                for (var i = 0; i < logits.Length; i++)
                {
                    logits[i] = (float)(rng.NextDouble() * 8.0);
                }
                Assert.InRange(m.Sample(logits), 0, logits.Length - 1);
            }
        }
    }
}
