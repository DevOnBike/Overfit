// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Sampling
{
    /// <summary>
    /// The zero-allocation top-nσ / typical-p survivor selection wired into <see cref="TokenSampler"/> must keep
    /// exactly the tokens the reference filters keep — verified through <see cref="TokenSampler.ComputeProbabilities"/>,
    /// which exposes the survivor set (survivors get non-zero probability, the rest 0).
    /// </summary>
    public sealed class TokenSamplerTruncatorTests
    {
        [Fact]
        public void TopNSigma_KeepsOnlyTheOutlierPeak()
        {
            var logits = new[] { 10f, 0f, 0f, 0f };
            var probs = ComputeProbs(logits, SamplingOptions.WithTopNSigma(1f, temperature: 1f));

            Assert.True(probs[0] > 0f);
            Assert.Equal(0f, probs[1]);
            Assert.Equal(0f, probs[2]);
            Assert.Equal(0f, probs[3]);
        }

        [Fact]
        public void TypicalP_OnUniform_KeepsCumulativeMassP()
        {
            var logits = new float[10]; // uniform
            var probs = ComputeProbs(logits, SamplingOptions.WithTypicalP(0.5f, temperature: 1f));

            var kept = 0;
            foreach (var p in probs)
            {
                if (p > 0f)
                {
                    kept++;
                }
            }
            Assert.Equal(5, kept); // 5 × 0.1 = 0.5
        }

        [Fact]
        public void TopNSigma_IsZeroAllocation()
        {
            var logits = new float[512];
            var rng = new Random(1);
            for (var i = 0; i < logits.Length; i++)
            {
                logits[i] = (float)(rng.NextDouble() * 8.0);
            }
            var idx = new int[logits.Length];
            var score = new float[logits.Length];
            var random = new Random(2);
            var opts = SamplingOptions.WithTopNSigma(1f, temperature: 0.8f);

            TokenSampler.Sample(logits, in opts, random, idx, score); // warm (JIT)

            var before = GC.GetAllocatedBytesForCurrentThread();
            for (var i = 0; i < 200; i++)
            {
                TokenSampler.Sample(logits, in opts, random, idx, score);
            }
            var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated < 1024, $"expected ~0 alloc, got {allocated} B over 200 calls");
        }

        private static float[] ComputeProbs(float[] logits, SamplingOptions opts)
        {
            var idx = new int[logits.Length];
            var score = new float[logits.Length];
            var probs = new float[logits.Length];
            TokenSampler.ComputeProbabilities(logits, in opts, idx, score, probs);
            return probs;
        }
    }
}
