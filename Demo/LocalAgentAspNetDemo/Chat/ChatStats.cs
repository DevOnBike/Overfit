// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Demo.LocalAgent.Chat
{
    /// <summary>Per-generation statistics returned alongside a chat reply.</summary>
    public record ChatStats(
        int PromptTokens,
        int GeneratedTokens,
        double TokensPerSecond,
        long AllocatedBytes,
        bool UsedKeyValueCache);
}
