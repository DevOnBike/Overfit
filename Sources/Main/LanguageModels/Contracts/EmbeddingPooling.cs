// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    /// <summary>How to pool per-token hidden states into one embedding vector.</summary>
    public enum EmbeddingPooling
    {
        /// <summary>Mean over all token hidden states (sentence-embedding default).</summary>
        Mean = 0,

        /// <summary>The last token's hidden state (causal-LM "last-token" pooling).</summary>
        LastToken = 1
    }
}
