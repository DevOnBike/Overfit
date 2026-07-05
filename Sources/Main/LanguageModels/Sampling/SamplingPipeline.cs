// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Sampling
{
    /// <summary>
    /// A composable, extensible token sampler: a list of history-aware <see cref="ILogitProcessor"/>s
    /// (e.g. repetition penalty) followed by a list of stateless <see cref="ISamplerStep"/>s
    /// (temperature / top-k / top-p / min-p / custom), then a terminal temperature-softmax multinomial draw.
    /// The additive counterpart to the engine's default <c>TokenSampler</c> + <c>SamplingOptions</c> — for
    /// callers who want to assemble or extend a bespoke strategy.
    ///
    /// <code>
    /// var pipeline = new SamplingPipeline()
    ///     .Use(new SamplingPipeline.RepetitionPenalty(1.1f, contextSize: 64))
    ///     .Use(new SamplingPipeline.Temperature(0.7f))
    ///     .Use(new SamplingPipeline.TopP(0.9f));
    /// int token = pipeline.Sample(logits, history, rng);
    /// </code>
    ///
    /// Steps allocate small scratch buffers per call (sort / nucleus), so this is the opt-in customizable
    /// path, NOT the zero-allocation decode hot path — that remains <c>TokenSampler</c>. Apply
    /// <see cref="Temperature"/> before the filtering steps (top-p / min-p read the post-temperature
    /// distribution).
    /// </summary>
    public sealed class SamplingPipeline
    {
        private readonly List<ILogitProcessor> _processors = [];
        private readonly List<ISamplerStep> _steps = [];

        /// <summary>Appends a history-aware logit processor; returns this for chaining.</summary>
        public SamplingPipeline Use(ILogitProcessor processor)
        {
            _processors.Add(processor ?? throw new ArgumentNullException(nameof(processor)));
            return this;
        }

        /// <summary>Appends a stateless sampler step; returns this for chaining.</summary>
        public SamplingPipeline Use(ISamplerStep step)
        {
            _steps.Add(step ?? throw new ArgumentNullException(nameof(step)));
            return this;
        }

        /// <summary>
        /// Runs the processors then the steps on <paramref name="logits"/> (in place) and draws a token from
        /// the resulting distribution with <paramref name="random"/>. <paramref name="history"/> is the tokens
        /// generated so far (for the processors). Falls back to the surviving argmax if the distribution
        /// degenerates (all-masked / non-finite sum).
        /// </summary>
        public int Sample(Span<float> logits, ReadOnlySpan<int> history, Random random)
        {
            if (logits.IsEmpty)
            {
                throw new ArgumentException("Logits cannot be empty.", nameof(logits));
            }
            ArgumentNullException.ThrowIfNull(random);

            foreach (var processor in _processors)
            {
                processor.Process(logits, history);
            }
            foreach (var step in _steps)
            {
                step.Apply(logits);
            }

            var maxIndex = 0;
            var max = float.NegativeInfinity;
            for (var i = 0; i < logits.Length; i++)
            {
                if (logits[i] > max)
                {
                    max = logits[i];
                    maxIndex = i;
                }
            }
            if (float.IsNegativeInfinity(max))
            {
                return 0;
            }   // everything masked — degenerate

            var sum = 0.0;
            for (var i = 0; i < logits.Length; i++)
            {
                if (!float.IsNegativeInfinity(logits[i]))
                {
                    sum += Math.Exp(logits[i] - max);
                }
            }
            if (sum <= 0.0 || double.IsNaN(sum) || double.IsInfinity(sum))
            {
                return maxIndex;
            }

            var target = random.NextDouble() * sum;
            var cumulative = 0.0;
            var last = maxIndex;
            for (var i = 0; i < logits.Length; i++)
            {
                if (float.IsNegativeInfinity(logits[i]))
                {
                    continue;
                }
                cumulative += Math.Exp(logits[i] - max);
                last = i;
                if (target <= cumulative)
                {
                    return i;
                }
            }
            return last;
        }

        // ── Built-in steps / processors (mirror TokenSampler / SamplingOptions semantics) ──────────────

        /// <summary>Scales finite logits by 1/temperature (sharpens for &lt;1, flattens for &gt;1).</summary>
        public sealed class Temperature : ISamplerStep
        {
            private readonly float _inverse;

            public Temperature(float temperature) => _inverse = 1f / MathF.Max(temperature, 1e-6f);

            public void Apply(Span<float> logits)
            {
                if (_inverse == 1f)
                {
                    return;
                }
                for (var i = 0; i < logits.Length; i++)
                {
                    if (!float.IsNegativeInfinity(logits[i]))
                    {
                        logits[i] *= _inverse;
                    }
                }
            }
        }

        /// <summary>Keeps the <c>k</c> highest-logit tokens; masks the rest.</summary>
        public sealed class TopK : ISamplerStep
        {
            private readonly int _k;

            public TopK(int k) => _k = k;

            public void Apply(Span<float> logits)
            {
                var n = logits.Length;
                if (_k <= 0 || _k >= n)
                {
                    return;
                }

                var copy = new float[n];
                logits.CopyTo(copy);
                Array.Sort(copy);                  // ascending
                var threshold = copy[n - _k];      // the k-th largest logit
                for (var i = 0; i < n; i++)
                {
                    if (logits[i] < threshold)
                    {
                        logits[i] = float.NegativeInfinity;
                    }
                }
            }
        }

        /// <summary>Nucleus sampling: keeps the smallest set of tokens whose probability mass ≥ <c>p</c>.</summary>
        public sealed class TopP : ISamplerStep
        {
            private readonly float _p;

            public TopP(float p) => _p = p;

            public void Apply(Span<float> logits)
            {
                if (_p >= 1f)
                {
                    return;
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
                    return;
                }

                var prob = new double[n];
                var idx = new int[n];
                var sum = 0.0;
                for (var i = 0; i < n; i++)
                {
                    var e = float.IsNegativeInfinity(logits[i]) ? 0.0 : Math.Exp(logits[i] - max);
                    prob[i] = e;
                    idx[i] = i;
                    sum += e;
                }
                if (sum <= 0.0)
                {
                    return;
                }
                for (var i = 0; i < n; i++)
                {
                    prob[i] /= sum;
                }

                Array.Sort(prob, idx);             // ascending by probability
                var keep = new bool[n];
                var cumulative = 0.0;
                for (var i = n - 1; i >= 0; i--)
                {
                    keep[idx[i]] = true;
                    cumulative += prob[i];
                    if (cumulative >= _p)
                    {
                        break;
                    }
                }
                for (var i = 0; i < n; i++)
                {
                    if (!keep[i])
                    {
                        logits[i] = float.NegativeInfinity;
                    }
                }
            }
        }

        /// <summary>Min-P: keeps tokens whose probability ≥ <c>minP × P(top)</c>. Run after
        /// <see cref="Temperature"/> (the threshold reads the post-temperature distribution).</summary>
        public sealed class MinP : ISamplerStep
        {
            private readonly float _minP;

            public MinP(float minP) => _minP = minP;

            public void Apply(Span<float> logits)
            {
                if (_minP <= 0f)
                {
                    return;
                }

                var max = float.NegativeInfinity;
                for (var i = 0; i < logits.Length; i++)
                {
                    if (logits[i] > max)
                    {
                        max = logits[i];
                    }
                }
                if (float.IsNegativeInfinity(max))
                {
                    return;
                }

                // P(token) ≥ minP·P(top) ⇔ logit ≥ maxLogit + ln(minP).
                var threshold = max + MathF.Log(Math.Clamp(_minP, 1e-6f, 1f));
                for (var i = 0; i < logits.Length; i++)
                {
                    if (logits[i] < threshold)
                    {
                        logits[i] = float.NegativeInfinity;
                    }
                }
            }
        }

        /// <summary>HuggingFace-style repetition penalty over the recent history (last
        /// <c>contextSize</c> tokens, or all when 0). No-op for penalty ≤ 1.</summary>
        public sealed class RepetitionPenalty : ILogitProcessor
        {
            private readonly float _penalty;
            private readonly int _contextSize;

            public RepetitionPenalty(float penalty, int contextSize = 0)
            {
                _penalty = penalty;
                _contextSize = contextSize;
            }

            public void Process(Span<float> logits, ReadOnlySpan<int> history)
            {
                if (_penalty <= 1f || history.IsEmpty)
                {
                    return;
                }

                var start = _contextSize > 0 && history.Length > _contextSize ? history.Length - _contextSize : 0;
                for (var i = start; i < history.Length; i++)
                {
                    var token = history[i];
                    if (token < 0 || token >= logits.Length)
                    {
                        continue;
                    }
                    var logit = logits[token];
                    logits[token] = logit < 0f ? logit * _penalty : logit / _penalty;
                }
            }
        }

        /// <summary>Top-nσ: keep only tokens whose logit ≥ <c>max − n·σ</c>, where σ is the standard deviation of
        /// the (finite) logits. A scale-adaptive truncation that acts directly on the pre-softmax logits —
        /// tight when the distribution is peaked, wide when flat — so it is robust to temperature. Run BEFORE
        /// <see cref="Temperature"/> (it reads the raw logits). n≈1 is typical; n≤0 is a no-op.</summary>
        public sealed class TopNSigma : ISamplerStep
        {
            private readonly float _n;

            public TopNSigma(float n) => _n = n;

            public void Apply(Span<float> logits)
            {
                if (_n <= 0f)
                {
                    return;
                }

                var count = 0;
                var max = float.NegativeInfinity;
                var sum = 0.0;
                for (var i = 0; i < logits.Length; i++)
                {
                    var v = logits[i];
                    if (float.IsNegativeInfinity(v))
                    {
                        continue;
                    }
                    count++;
                    sum += v;
                    if (v > max)
                    {
                        max = v;
                    }
                }
                if (count == 0)
                {
                    return;
                }

                var mean = sum / count;
                var varSum = 0.0;
                for (var i = 0; i < logits.Length; i++)
                {
                    var v = logits[i];
                    if (float.IsNegativeInfinity(v))
                    {
                        continue;
                    }
                    var d = v - mean;
                    varSum += d * d;
                }
                var sigma = Math.Sqrt(varSum / count);
                var threshold = max - (_n * sigma);
                for (var i = 0; i < logits.Length; i++)
                {
                    if (logits[i] < threshold)
                    {
                        logits[i] = float.NegativeInfinity;
                    }
                }
            }
        }

        /// <summary>Locally typical sampling: keep the tokens whose information content <c>−ln p</c> is closest
        /// to the distribution's entropy, until their cumulative probability ≥ <c>p</c>. Favours "typical"
        /// continuations (near the expected surprise) rather than merely the most probable — reduces both
        /// degenerate repetition and incoherent low-probability picks. p≈0.95 typical; p≥1 is a no-op. Run
        /// after <see cref="Temperature"/>.</summary>
        public sealed class TypicalP : ISamplerStep
        {
            private readonly float _p;

            public TypicalP(float p) => _p = p;

            public void Apply(Span<float> logits)
            {
                if (_p >= 1f)
                {
                    return;
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
                    return;
                }

                var prob = new double[n];
                var sum = 0.0;
                for (var i = 0; i < n; i++)
                {
                    var e = float.IsNegativeInfinity(logits[i]) ? 0.0 : Math.Exp(logits[i] - max);
                    prob[i] = e;
                    sum += e;
                }
                if (sum <= 0.0)
                {
                    return;
                }

                var entropy = 0.0;
                for (var i = 0; i < n; i++)
                {
                    prob[i] /= sum;
                    if (prob[i] > 0.0)
                    {
                        entropy -= prob[i] * Math.Log(prob[i]);
                    }
                }

                // Deviation of each token's surprise from the entropy; keep the smallest-deviation tokens.
                var dev = new double[n];
                var idx = new int[n];
                for (var i = 0; i < n; i++)
                {
                    dev[i] = prob[i] > 0.0 ? Math.Abs(-Math.Log(prob[i]) - entropy) : double.PositiveInfinity;
                    idx[i] = i;
                }
                Array.Sort(dev, idx);

                var keep = new bool[n];
                var cumulative = 0.0;
                for (var i = 0; i < n; i++)
                {
                    var t = idx[i];
                    keep[t] = true;
                    cumulative += prob[t];
                    if (cumulative >= _p)
                    {
                        break;
                    }
                }
                for (var i = 0; i < n; i++)
                {
                    if (!keep[i])
                    {
                        logits[i] = float.NegativeInfinity;
                    }
                }
            }
        }

        /// <summary>XTC (Exclude Top Choices): with the configured <c>probability</c>, drop every token above
        /// <c>threshold</c> EXCEPT the least-probable one — removing the "obvious" high-probability choices to
        /// boost variety while leaving the long tail untouched. Holds its own RNG (the pipeline's terminal draw
        /// is separate). threshold≈0.1, probability≈0.5. No-op when the roll fails or fewer than two tokens
        /// clear the threshold. Run after <see cref="Temperature"/>.</summary>
        public sealed class Xtc : ISamplerStep
        {
            private readonly float _threshold;
            private readonly float _probability;
            private readonly Random _random;

            public Xtc(float threshold, float probability, int seed = 0)
            {
                _threshold = threshold;
                _probability = probability;
                _random = new Random(seed);
            }

            public void Apply(Span<float> logits)
            {
                if (_probability <= 0f || _threshold > 0.5f)
                {
                    return;
                }
                if (_random.NextDouble() > _probability)
                {
                    return;
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
                    return;
                }

                var prob = new double[n];
                var sum = 0.0;
                for (var i = 0; i < n; i++)
                {
                    var e = float.IsNegativeInfinity(logits[i]) ? 0.0 : Math.Exp(logits[i] - max);
                    prob[i] = e;
                    sum += e;
                }
                if (sum <= 0.0)
                {
                    return;
                }

                // Collect tokens whose probability clears the threshold; keep the least-probable, mask the rest.
                var above = new List<int>();
                for (var i = 0; i < n; i++)
                {
                    if (prob[i] / sum >= _threshold)
                    {
                        above.Add(i);
                    }
                }
                if (above.Count < 2)
                {
                    return;
                }
                above.Sort((a, b) => prob[b].CompareTo(prob[a])); // descending probability
                for (var k = 0; k < above.Count - 1; k++)
                {
                    logits[above[k]] = float.NegativeInfinity;
                }
            }
        }

        /// <summary>DRY (Don't Repeat Yourself): penalises the token that would extend the longest verbatim
        /// repetition of the recent history, so loops are broken by their <em>length</em> rather than by
        /// blanket per-token penalties. For each candidate the length L of the repetition it would create is
        /// found with the Z-algorithm; tokens with L ≥ <c>allowedLength</c> lose
        /// <c>multiplier · base^(L − allowedLength)</c> from their logit. Typical: multiplier 0.8, base 1.75,
        /// allowedLength 2. penaltyLastN limits the window (0 = all history).</summary>
        public sealed class Dry : ILogitProcessor
        {
            private const int MaxExponent = 32; // cap base^(…) so a long loop can't overflow the penalty
            private readonly float _multiplier;
            private readonly float _base;
            private readonly int _allowedLength;
            private readonly int _penaltyLastN;

            public Dry(float multiplier = 0.8f, float @base = 1.75f, int allowedLength = 2, int penaltyLastN = 0)
            {
                _multiplier = multiplier;
                _base = @base;
                _allowedLength = Math.Max(1, allowedLength);
                _penaltyLastN = penaltyLastN;
            }

            public void Process(Span<float> logits, ReadOnlySpan<int> history)
            {
                if (_multiplier <= 0f || history.Length < 2)
                {
                    return;
                }

                var start = _penaltyLastN > 0 && history.Length > _penaltyLastN ? history.Length - _penaltyLastN : 0;
                var window = history[start..];
                var m = window.Length;
                if (m < 2)
                {
                    return;
                }


                // Opt-in (pipeline) path allocates its Z-algorithm scratch; the zero-alloc engine hot path calls
                // Apply directly with caller-owned rev/z buffers.
                var rev = new int[m];
                var z = new int[m];
                Apply(logits, window, _multiplier, _base, _allowedLength, rev, z);
            }

            /// <summary>Zero-allocation DRY core: penalises the token that would extend the longest verbatim
            /// repetition of <paramref name="window"/> (the recent token history). <paramref name="rev"/> and
            /// <paramref name="z"/> are caller-owned scratch (each length ≥ window length) — so the decode hot
            /// path can reuse fixed buffers instead of allocating per token. Both scratch spans are fully
            /// (re)written here, so they may be dirty on entry.</summary>
            internal static void Apply(
                Span<float> logits,
                ReadOnlySpan<int> window,
                float multiplier,
                float baseValue,
                int allowedLength,
                Span<int> rev,
                Span<int> z)
            {
                if (multiplier <= 0f)
                {
                    return;
                }
                var m = window.Length;
                if (m < 2 || rev.Length < m || z.Length < m)
                {
                    return;
                }
                allowedLength = Math.Max(1, allowedLength);

                // Z-algorithm over the reversed window: rev[i]'s match length against the prefix tells us how
                // long the current suffix repeats an earlier occurrence ending at window position m-1-i; the
                // token that FOLLOWED that occurrence is window[m-i], the continuation we penalise.
                for (var i = 0; i < m; i++)
                {
                    rev[i] = window[m - 1 - i];
                }

                z[0] = 0;
                var l = 0;
                var r = 0;
                for (var i = 1; i < m; i++)
                {
                    z[i] = 0;
                    if (i < r)
                    {
                        z[i] = Math.Min(r - i, z[i - l]);
                    }
                    while (i + z[i] < m && rev[z[i]] == rev[i + z[i]])
                    {
                        z[i]++;
                    }
                    if (i + z[i] > r)
                    {
                        l = i;
                        r = i + z[i];
                    }
                }

                for (var i = 1; i < m; i++)
                {
                    var matchLen = z[i];
                    if (matchLen < allowedLength)
                    {
                        continue;
                    }
                    var candidate = window[m - i];
                    if (candidate < 0 || candidate >= logits.Length || float.IsNegativeInfinity(logits[candidate]))
                    {
                        continue;
                    }
                    var exponent = Math.Min(matchLen - allowedLength, MaxExponent);
                    var penalty = multiplier * MathF.Pow(baseValue, exponent);
                    logits[candidate] -= penalty;
                }
            }
        }
    }
}
