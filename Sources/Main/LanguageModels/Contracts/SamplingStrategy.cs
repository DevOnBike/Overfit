// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public enum SamplingStrategy
    {
        Greedy = 0,
        Temperature = 1,
        TopK = 2,
        TopP = 3,
        TopKTopP = 4,

        /// <summary>
        /// Min-P: keep only tokens whose probability is at least <c>MinP × P(most-likely)</c>,
        /// then sample (with temperature) from the survivors. A scale-adaptive alternative to
        /// Top-P that widens on confident steps and narrows on flat ones.
        /// </summary>
        MinP = 5,

        /// <summary>
        /// Top-nσ: keep tokens whose logit ≥ <c>max − NSigma·σ</c> (σ = std-dev of the logits). Acts on the
        /// raw pre-softmax logits, so it stays stable across temperature — a strong, scale-adaptive truncator
        /// that trims the low-probability tail where hallucinations live.
        /// </summary>
        TopNSigma = 6,

        /// <summary>
        /// Locally typical sampling: keep the tokens whose surprise (<c>−ln p</c>) is closest to the
        /// distribution's entropy, until their cumulative probability ≥ <c>TypicalP</c>. Favours "typical"
        /// continuations over merely the most probable ones.
        /// </summary>
        TypicalP = 7
    }
}
