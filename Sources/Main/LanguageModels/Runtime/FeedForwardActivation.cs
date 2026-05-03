// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Activation used by the cached single-token feed-forward decode block.
    /// </summary>
    public enum FeedForwardActivation
    {
        None = 0,
        ReLU = 1,
        GeLU = 2
    }
}
