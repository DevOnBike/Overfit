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
    }
}
