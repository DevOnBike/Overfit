// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// The rejection-sampling core of sampling-correct speculative decoding for a <b>point-mass</b>
    /// (prompt-lookup / n-gram) draft. The draft proposal is deterministic, so the speculative-sampling
    /// rule degenerates to: accept the drafted token <c>d</c> with probability <c>p(d)</c> under the
    /// target distribution <c>p</c>; otherwise sample from the renormalised residual
    /// <c>norm(max(0, p − e_d))</c>. The resulting token is distributed <b>exactly</b> as a direct draw
    /// from <c>p</c> (proof: P(accept d)·1 + P(reject)·residual reproduces p for every token), so the
    /// generated sequence has the same distribution as non-speculative sampling — verified statistically
    /// in <c>SpeculativeSamplerTests</c>.
    /// </summary>
    public static class SpeculativeSampler
    {
        /// <summary>
        /// Accepts <paramref name="draft"/> with probability <c>p[draft]</c>, else returns a token sampled
        /// from the residual. The returned token equals <paramref name="draft"/> iff it was accepted (the
        /// residual zeroes the draft, so a resample never returns it). <paramref name="probabilities"/> is
        /// not mutated; <paramref name="residualScratch"/> (≥ its length) holds the residual.
        /// </summary>
        public static int AcceptOrResample(
            ReadOnlySpan<float> probabilities, int draft, Random random, Span<float> residualScratch)
        {
            ArgumentNullException.ThrowIfNull(random);
            if ((uint)draft >= (uint)probabilities.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(draft));
            }

            if (random.NextSingle() < probabilities[draft])
            {
                return draft;
            }

            probabilities.CopyTo(residualScratch);
            residualScratch[draft] = 0f;   // residual: norm(max(0, p − e_draft))
            return Sample(residualScratch[..probabilities.Length], random);
        }

        /// <summary>Samples a token index from a (sub-)distribution span; falls back to the argmax if the mass is ≤ 0.</summary>
        public static int Sample(ReadOnlySpan<float> probabilities, Random random)
        {
            ArgumentNullException.ThrowIfNull(random);

            var sum = 0.0;
            for (var i = 0; i < probabilities.Length; i++) { sum += probabilities[i]; }
            if (sum <= 0.0) { return ArgMax(probabilities); }

            var u = random.NextDouble() * sum;
            var cumulative = 0.0;
            for (var i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (u <= cumulative) { return i; }
            }
            return probabilities.Length - 1;
        }

        private static int ArgMax(ReadOnlySpan<float> values)
        {
            var best = 0;
            for (var i = 1; i < values.Length; i++)
            {
                if (values[i] > values[best]) { best = i; }
            }
            return best;
        }
    }
}
