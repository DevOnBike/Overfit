// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Sampling
{
    /// <summary>
    /// Mirostat — a feedback sampler that holds the <em>perplexity</em> of the output near a target instead of
    /// fixing a truncation like top-k / top-p. Each step it truncates the distribution to the surprise budget
    /// <c>μ</c>, samples a token, then nudges <c>μ</c> by the error between the token's observed surprise and
    /// the target <c>τ</c> (learning rate <c>η</c>). This keeps long generations from drifting into repetition
    /// (surprise collapsing) or incoherence (surprise exploding). Stateful — one instance per generation stream.
    ///
    /// <para><see cref="Version.V2"/> is the simple, robust variant (direct surprise truncation); <see cref="Version.V1"/>
    /// additionally estimates the distribution's power-law exponent to pick a top-k. Terminal sampler: it draws
    /// the token itself (unlike a stateless <see cref="ISamplerStep"/>), so use it in place of a pipeline's
    /// final draw, after any logit processors have run.</para>
    /// </summary>
    public sealed class MirostatSampler
    {
        public enum Version
        {
            V1 = 1,
            V2 = 2,
        }

        private readonly Version _version;
        private readonly double _tau;
        private readonly double _eta;
        private readonly int _estimateWindow;
        private readonly Random _random;
        private double _mu;

        /// <param name="tau">Target surprise (bits) — lower = more focused, higher = more diverse. ~5 is typical.</param>
        /// <param name="eta">Learning rate for the μ feedback. 0.1 is typical.</param>
        /// <param name="version">V2 (default, robust) or V1 (Zipf-exponent estimate → top-k).</param>
        /// <param name="seed">RNG seed for the multinomial draw.</param>
        /// <param name="estimateWindow">V1 only: how many leading tokens estimate the power-law exponent.</param>
        public MirostatSampler(
            float tau = 5.0f, float eta = 0.1f, Version version = Version.V2, int seed = 0, int estimateWindow = 100)
        {
            _version = version;
            _tau = tau;
            _eta = eta;
            _estimateWindow = Math.Max(2, estimateWindow);
            _random = new Random(seed);
            _mu = 2.0 * tau;
        }

        /// <summary>Current surprise budget μ (starts at 2τ, adapts each <see cref="Sample"/>). Exposed for
        /// inspection / tests.</summary>
        public double Mu => _mu;

        /// <summary>Draws the next token from <paramref name="logits"/> and updates μ. Does not modify the
        /// logits. Returns 0 if the distribution is degenerate (all-masked / non-finite).</summary>
        public int Sample(Span<float> logits)
        {
            if (logits.IsEmpty)
            {
                throw new ArgumentException("Logits cannot be empty.", nameof(logits));
            }

            var n = logits.Length;
            var max = float.NegativeInfinity;
            for (var i = 0; i < n; i++)
            {
                if (logits[i] > max)
                {
                    max = logits[i];
                }
            }
            if (float.IsNegativeInfinity(max))
            {
                return 0;
            }

            // Softmax → probabilities, sorted descending (ascending negative-prob keeps the token indices).
            var negProb = new double[n];
            var idx = new int[n];
            var sum = 0.0;
            for (var i = 0; i < n; i++)
            {
                var e = float.IsNegativeInfinity(logits[i]) ? 0.0 : Math.Exp(logits[i] - max);
                negProb[i] = e;
                idx[i] = i;
                sum += e;
            }
            if (sum <= 0.0)
            {
                return 0;
            }
            for (var i = 0; i < n; i++)
            {
                negProb[i] = -(negProb[i] / sum); // negated normalised prob → ascending sort = descending prob
            }
            Array.Sort(negProb, idx);

            var keep = _version == Version.V2
                ? TruncateBySurpriseV2(negProb, n)
                : TruncateByTopKV1(negProb, n);

            // Multinomial draw within the kept prefix.
            var keptMass = 0.0;
            for (var k = 0; k < keep; k++)
            {
                keptMass += -negProb[k];
            }
            var chosenPos = keep - 1;
            if (keptMass > 0.0)
            {
                var target = _random.NextDouble() * keptMass;
                var cumulative = 0.0;
                for (var k = 0; k < keep; k++)
                {
                    cumulative += -negProb[k];
                    if (target <= cumulative)
                    {
                        chosenPos = k;
                        break;
                    }
                }
            }

            var chosenProb = -negProb[chosenPos];
            var token = idx[chosenPos];

            // Feedback: μ ← μ − η·(observed surprise − τ).
            var observedSurprise = chosenProb > 0.0 ? -Math.Log2(chosenProb) : _tau;
            _mu -= _eta * (observedSurprise - _tau);
            return token;
        }

        // V2: keep the leading tokens whose surprise (−log2 p) stays within the μ budget (at least one).
        private int TruncateBySurpriseV2(double[] negProbAscending, int n)
        {
            var keep = 0;
            for (var k = 0; k < n; k++)
            {
                var p = -negProbAscending[k];
                if (p <= 0.0)
                {
                    break;
                }
                var surprise = -Math.Log2(p);
                if (surprise > _mu && keep >= 1)
                {
                    break;
                }
                keep = k + 1;
            }
            return Math.Max(1, keep);
        }

        // V1: estimate the Zipf exponent ŝ from the leading probability ratios, derive a k from μ, top-k it.
        private int TruncateByTopKV1(double[] negProbAscending, int n)
        {
            var window = Math.Min(_estimateWindow, n - 1);
            if (window < 1)
            {
                return Math.Max(1, n);
            }

            var sumTi2 = 0.0;
            var sumTiBi = 0.0;
            for (var i = 0; i < window; i++)
            {
                var pi = -negProbAscending[i];
                var pi1 = -negProbAscending[i + 1];
                if (pi <= 0.0 || pi1 <= 0.0)
                {
                    break;
                }
                var ti = Math.Log((i + 2.0) / (i + 1.0));
                var bi = Math.Log(pi / pi1);
                sumTi2 += ti * ti;
                sumTiBi += ti * bi;
            }
            if (sumTi2 <= 0.0)
            {
                return Math.Max(1, n);
            }

            var sHat = sumTiBi / sumTi2;
            var epsilonHat = sHat - 1.0;
            var pow2Mu = Math.Pow(2.0, _mu);
            var denom = 1.0 - Math.Pow(n, -epsilonHat);
            if (Math.Abs(denom) < 1e-9 || sHat == 0.0)
            {
                return Math.Max(1, n);
            }
            var kEstimate = Math.Pow(epsilonHat * pow2Mu / denom, 1.0 / sHat);
            if (double.IsNaN(kEstimate) || double.IsInfinity(kEstimate))
            {
                return Math.Max(1, n);
            }
            var k = (int)Math.Round(kEstimate);
            return Math.Clamp(k, 1, n);
        }
    }
}
