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
        MinP = 5
    }
}
