// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// GPT1Config extensions for modern SLM architectures (GQA, RoPE, SwiGLU).
    /// </summary>
    public static class LlamaConfig
    {
        // ── Llama 3.2 ────────────────────────────────────────────────────────

        /// <summary>
        /// Llama 3.2 1B — Meta, 2024.
        /// GQA: 32 Q heads / 8 KV heads → KV cache 4× smaller than MHA.
        /// RoPE theta = 500_000 (extended context).
        /// </summary>
        public static GPT1Config Llama32_1B => new()
        {
            VocabSize = 128_256,
            ContextLength = 8_192,    // practical for CPU; model supports 131072
            DModel = 2_048,
            NHeads = 32,
            NKvHeads = 8,
            NLayers = 16,
            DFF = 8_192,
            TieWeights = true,
            PreLayerNorm = true,
            FfnActivation = FeedForwardActivation.SwiGLU,
            UseRoPE = true,
            RoPETheta = 500_000f,
        };

        /// <summary>
        /// Llama 3.2 3B — Meta, 2024.
        /// GQA: 24 Q heads / 8 KV heads.
        /// </summary>
        public static GPT1Config Llama32_3B => new()
        {
            VocabSize = 128_256,
            ContextLength = 8_192,
            DModel = 3_072,
            NHeads = 24,
            NKvHeads = 8,
            NLayers = 28,
            DFF = 8_192,
            TieWeights = true,
            PreLayerNorm = true,
            FfnActivation = FeedForwardActivation.SwiGLU,
            UseRoPE = true,
            RoPETheta = 500_000f,
        };

        // ── Phi-3 ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Phi-3-mini 3.8B — Microsoft, 2024.
        /// MHA (no GQA), RoPE, SwiGLU.
        /// </summary>
        public static GPT1Config Phi3Mini => new()
        {
            VocabSize = 32_064,
            ContextLength = 4_096,
            DModel = 3_072,
            NHeads = 32,
            NKvHeads = 32,   // MHA — no GQA
            NLayers = 32,
            DFF = 8_192,
            TieWeights = false,
            PreLayerNorm = true,
            FfnActivation = FeedForwardActivation.SwiGLU,
            UseRoPE = true,
            RoPETheta = 10_000f,
        };

        // ── Qwen 2.5 ──────────────────────────────────────────────────────────

        /// <summary>
        /// Qwen 2.5 0.5B — Alibaba, 2024.
        /// GQA: 14 Q heads / 2 KV heads.
        /// </summary>
        public static GPT1Config Qwen25_0_5B => new()
        {
            VocabSize = 151_936,
            ContextLength = 4_096,
            DModel = 896,
            NHeads = 14,
            NKvHeads = 2,
            NLayers = 24,
            DFF = 4_864,
            TieWeights = true,
            PreLayerNorm = true,
            FfnActivation = FeedForwardActivation.SwiGLU,
            UseRoPE = true,
            RoPETheta = 1_000_000f,
        };

        /// <summary>
        /// Qwen 2.5 1.5B — Alibaba, 2024.
        /// </summary>
        public static GPT1Config Qwen25_1_5B => new()
        {
            VocabSize = 151_936,
            ContextLength = 4_096,
            DModel = 1_536,
            NHeads = 12,
            NKvHeads = 2,
            NLayers = 28,
            DFF = 8_960,
            TieWeights = true,
            PreLayerNorm = true,
            FfnActivation = FeedForwardActivation.SwiGLU,
            UseRoPE = true,
            RoPETheta = 1_000_000f,
        };
    }
}
