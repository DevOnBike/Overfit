// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// GPT-2 model configurations matching the original OpenAI release.
    ///
    /// GPT-2 uses the same transformer architecture as GPT-1 (decoder-only)
    /// with two differences:
    ///   1. Pre-LayerNorm instead of post-LayerNorm
    ///   2. Larger vocab (50257 BPE tokens vs GPT-1's 40478)
    ///
    /// Both are already supported by GPT1Model + GPT1Config.
    /// These presets map directly to GPT1Config instances.
    ///
    /// Usage:
    ///   using var model = new GPT1Model(Gpt2Config.Small);
    ///   model.Load(new BinaryReader(File.OpenRead("gpt2_small.bin")));
    ///
    /// Weights: convert_gpt2.py (Scripts/) downloads and converts
    /// from HuggingFace to Overfit binary format.
    ///
    /// Tokenizer:
    ///   var tokenizer = BytePairEncoder.Load("vocab.json", "merges.txt");
    ///   // vocab.json + merges.txt from: huggingface.co/openai-community/gpt2
    /// </summary>
    public static class Gpt2Config
    {
        /// <summary>
        /// GPT-2 Small — 117M parameters.
        /// 12 layers, 768d, 12 heads, 3072 FFN, 1024 context, 50257 vocab.
        /// Checkpoint: ~450MB (gpt2_small.bin)
        /// </summary>
        public static GPT1Config Small => new()
        {
            VocabSize     = 50257,
            ContextLength = 1024,
            DModel        = 768,
            NHeads        = 12,
            NLayers       = 12,
            DFF           = 3072,
            TieWeights    = false,  // Overfit stores LMHead separately after conversion
            PreLayerNorm  = true,   // GPT-2 uses pre-LN
        };

        /// <summary>
        /// GPT-2 Medium — 345M parameters.
        /// 24 layers, 1024d, 16 heads, 4096 FFN, 1024 context, 50257 vocab.
        /// Checkpoint: ~1.3GB (gpt2_medium.bin)
        /// </summary>
        public static GPT1Config Medium => new()
        {
            VocabSize     = 50257,
            ContextLength = 1024,
            DModel        = 1024,
            NHeads        = 16,
            NLayers       = 24,
            DFF           = 4096,
            TieWeights    = false,
            PreLayerNorm  = true,
        };

        /// <summary>
        /// GPT-2 Large — 762M parameters.
        /// 36 layers, 1280d, 20 heads, 5120 FFN, 1024 context, 50257 vocab.
        /// Checkpoint: ~3GB (gpt2_large.bin)
        /// </summary>
        public static GPT1Config Large => new()
        {
            VocabSize     = 50257,
            ContextLength = 1024,
            DModel        = 1280,
            NHeads        = 20,
            NLayers       = 36,
            DFF           = 5120,
            TieWeights    = false,
            PreLayerNorm  = true,
        };

        /// <summary>
        /// GPT-2 XL — 1.5B parameters.
        /// 48 layers, 1600d, 25 heads, 6400 FFN, 1024 context, 50257 vocab.
        /// Checkpoint: ~6GB (gpt2_xl.bin)
        /// Requires: ComputationGraph arena ≥ 4GB for inference.
        /// </summary>
        public static GPT1Config XL => new()
        {
            VocabSize     = 50257,
            ContextLength = 1024,
            DModel        = 1600,
            NHeads        = 25,
            NLayers       = 48,
            DFF           = 6400,
            TieWeights    = false,
            PreLayerNorm  = true,
        };
    }
}
