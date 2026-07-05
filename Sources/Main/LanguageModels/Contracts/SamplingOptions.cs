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

        /// <summary>
        /// Top-nσ sampling (default n 1.0): keep tokens with logit ≥ <c>max − n·σ</c>, sample with
        /// <paramref name="temperature"/> from the survivors. A strong, scale-adaptive tail-trimmer — good for
        /// reducing hallucinations while keeping some diversity. Selects <see cref="SamplingStrategy.TopNSigma"/>.
        /// </summary>
        public static SamplingOptions WithTopNSigma(float nSigma = 1.0f, float temperature = 1.0f, int seed = 0)
        {
            return new SamplingOptions(
                strategy: SamplingStrategy.TopNSigma,
                temperature: temperature,
                topK: 0,
                topP: 1.0f,
                seed: seed,
                nSigma: nSigma);
        }

        /// <summary>
        /// Locally typical sampling (default p 0.95): keep the tokens whose surprise is closest to the entropy
        /// until their cumulative probability ≥ <paramref name="typicalP"/>, sample with
        /// <paramref name="temperature"/>. Selects <see cref="SamplingStrategy.TypicalP"/>.
        /// </summary>
        public static SamplingOptions WithTypicalP(float typicalP = 0.95f, float temperature = 1.0f, int seed = 0)
        {
            return new SamplingOptions(
                strategy: SamplingStrategy.TypicalP,
                temperature: temperature,
                topK: 0,
                topP: 1.0f,
                seed: seed,
                typicalP: typicalP);
        }

        public SamplingOptions(
            SamplingStrategy strategy,
            float temperature,
            int topK,
            float topP,
            int seed,
            float repetitionPenalty = 1.0f,
            int repetitionPenaltyContextSize = 0,
            float minP = 0f,
            float nSigma = 0f,
            float typicalP = 1f,
            float dryMultiplier = 0f,
            float dryBase = 1.75f,
            int dryAllowedLength = 2,
            int dryPenaltyLastN = 0)
        {
            Strategy = strategy;
            Temperature = temperature;
            TopK = topK;
            TopP = topP;
            Seed = seed;
            RepetitionPenalty = repetitionPenalty;
            RepetitionPenaltyContextSize = repetitionPenaltyContextSize;
            MinP = minP;
            NSigma = nSigma;
            TypicalP = typicalP;
            DryMultiplier = dryMultiplier;
            DryBase = dryBase;
            DryAllowedLength = dryAllowedLength;
            DryPenaltyLastN = dryPenaltyLastN;
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

        /// <summary>
        /// Top-nσ multiplier: a token survives if its logit ≥ <c>max − NSigma·σ</c>. Used by
        /// <see cref="SamplingStrategy.TopNSigma"/>. Typical: 1.0. Larger = wider.
        /// </summary>
        public float NSigma
        {
            get;
        }

        /// <summary>
        /// Locally-typical cumulative-probability threshold ∈ (0, 1]. Used by
        /// <see cref="SamplingStrategy.TypicalP"/>. 1 = disabled. Typical: 0.95.
        /// </summary>
        public float TypicalP
        {
            get;
        }

        /// <summary>
        /// DRY (Don't Repeat Yourself) penalty weight. When &gt; 0 the engine subtracts
        /// <c>DryMultiplier · DryBase^(L − DryAllowedLength)</c> from any token that would extend a verbatim
        /// repetition of length <c>L ≥ DryAllowedLength</c> in the recent output. Orthogonal to the sampling
        /// strategy — applied to the logits BEFORE selection, so it works even under greedy decode. 0 = disabled.
        /// Typical: 0.8.
        /// </summary>
        public float DryMultiplier
        {
            get;
        }

        /// <summary>
        /// DRY penalty growth base (per repetition character beyond <see cref="DryAllowedLength"/>). Typical: 1.75.
        /// </summary>
        public float DryBase
        {
            get;
        }

        /// <summary>
        /// Repetition length that DRY tolerates before penalising (matches shorter than this are free). Typical: 2.
        /// </summary>
        public int DryAllowedLength
        {
            get;
        }

        /// <summary>
        /// Window of recent output tokens DRY scans (0 = all tokens generated since Reset). Typical: 256.
        /// </summary>
        public int DryPenaltyLastN
        {
            get;
        }
    }
}
