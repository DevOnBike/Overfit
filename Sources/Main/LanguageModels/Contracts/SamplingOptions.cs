// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public readonly struct SamplingOptions
    {
        public static SamplingOptions Greedy
        {
            get;
        } = new(
            strategy: SamplingStrategy.Greedy,
            temperature: 1.0f,
            topK: 0,
            topP: 1.0f,
            seed: 0,
            repetitionPenalty: 1.0f,
            repetitionPenaltyContextSize: 0);

        /// <summary>
        /// Greedy sampling with repetition penalty (default 1.1 — HuggingFace default).
        /// Use this to prevent generation loops without losing determinism.
        /// </summary>
        public static SamplingOptions GreedyWithPenalty(
            float penalty = 1.1f,
            int contextSize = 64)
        {
            return new SamplingOptions(
                strategy: SamplingStrategy.Greedy,
                temperature: 1.0f,
                topK: 0,
                topP: 1.0f,
                seed: 0,
                repetitionPenalty: penalty,
                repetitionPenaltyContextSize: contextSize);
        }

        /// <summary>
        /// Min-P sampling (default minP 0.05): keep tokens with probability ≥ minP × P(top),
        /// sample with <paramref name="temperature"/> from survivors. minP &gt; 0 selects the
        /// <see cref="SamplingStrategy.MinP"/> strategy.
        /// </summary>
        public static SamplingOptions WithMinP(float minP = 0.05f, float temperature = 1.0f, int seed = 0)
        {
            return new SamplingOptions(
                strategy: SamplingStrategy.MinP,
                temperature: temperature,
                topK: 0,
                topP: 1.0f,
                seed: seed,
                minP: minP);
        }

        public SamplingOptions(
            SamplingStrategy strategy,
            float temperature,
            int topK,
            float topP,
            int seed,
            float repetitionPenalty = 1.0f,
            int repetitionPenaltyContextSize = 0,
            float minP = 0f)
        {
            Strategy = strategy;
            Temperature = temperature;
            TopK = topK;
            TopP = topP;
            Seed = seed;
            RepetitionPenalty = repetitionPenalty;
            RepetitionPenaltyContextSize = repetitionPenaltyContextSize;
            MinP = minP;
        }

        public SamplingStrategy Strategy
        {
            get;
        }

        public float Temperature
        {
            get;
        }

        public int TopK
        {
            get;
        }

        public float TopP
        {
            get;
        }

        public int Seed
        {
            get;
        }

        /// <summary>
        /// Repetition penalty applied to logits of recently-used tokens.
        /// Standard formula (HuggingFace):
        ///   if logit &lt; 0: logit *= penalty
        ///   else:           logit /= penalty
        /// Default 1.0 = disabled. Typical values: 1.1 (moderate), 1.3 (aggressive).
        /// Penalty &lt;= 1.0 disables the feature.
        /// </summary>
        public float RepetitionPenalty
        {
            get;
        }

        /// <summary>
        /// Window of recent tokens considered for repetition penalty.
        /// 0 = unlimited (all tokens since Reset). Positive N = last N tokens only.
        /// Smaller windows let the model repeat tokens after some distance.
        /// </summary>
        public int RepetitionPenaltyContextSize
        {
            get;
        }

        /// <summary>
        /// Min-P threshold ∈ (0, 1): a token survives if its probability ≥ <c>MinP × P(top)</c>.
        /// Used by <see cref="SamplingStrategy.MinP"/>. 0 = disabled. Typical: 0.05–0.1.
        /// </summary>
        public float MinP
        {
            get;
        }
    }
}
