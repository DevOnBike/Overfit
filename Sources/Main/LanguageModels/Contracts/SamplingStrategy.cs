// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
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
        TopKTopP = 4
    }
}
