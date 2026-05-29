// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Demo.LocalAgent.Rag
{
    public record RagAnswer(
        string Reply,
        IReadOnlyList<RagSource> Sources,
        int PromptTokens,
        int GeneratedTokens,
        double TokensPerSecond,
        double SearchSeconds);
}