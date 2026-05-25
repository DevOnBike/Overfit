// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Hyperparameter configuration for GPT-style models.
    ///
    /// GPT-1 (Radford et al., 2018):
    /// VocabSize = 40478 (BPE vocabulary)
    /// ContextLength = 512
    /// DModel = 768
    /// NHeads = 12
    /// NLayers = 12
    /// DFF = 3072 (= 4 * DModel)
    ///
    /// GPT-2 Small (for reference):
    /// VocabSize = 50257
    /// ContextLength = 1024
    /// DModel = 768
    /// NHeads = 12
    /// NLayers = 12
    /// DFF = 3072
    /// </summary>
    public sealed class GPT1Config
    {
        /// <summary>GPT-1 original configuration.</summary>
        public static readonly GPT1Config GPT1 = new()
        {
            VocabSize = 40478,
            ContextLength = 512,
            DModel = 768,
            NHeads = 12,
            NLayers = 12,
            DFF = 3072,
        };

        /// <summary>Small config for testing (fits in memory, fast to instantiate).</summary>
        public static readonly GPT1Config Small = new()
        {
            VocabSize = 256,
            ContextLength = 16,
            DModel = 64,
            NHeads = 4,
            NLayers = 2,
            DFF = 256,
        };

        /// <summary>Vocabulary size (number of BPE tokens).</summary>
        public int VocabSize { get; init; } = 40478;

        /// <summary>Maximum sequence length (context window).</summary>
        public int ContextLength { get; init; } = 512;

        /// <summary>Model dimension (embedding size, d_model).</summary>
        public int DModel { get; init; } = 768;

        /// <summary>Number of attention heads per block.</summary>
        public int NHeads { get; init; } = 12;

        /// <summary>Number of Transformer blocks.</summary>
        public int NLayers { get; init; } = 12;

        /// <summary>Feed-forward inner dimension (typically 4 * DModel).</summary>
        public int DFF { get; init; } = 3072;

        /// <summary>Layer norm epsilon.</summary>
        public float LNEps { get; init; } = 1e-5f;

        /// <summary>
        /// Use Pre-LayerNorm (true, GPT-2 style, more stable) or
        /// Post-LayerNorm (false, original GPT-1).
        /// </summary>
        public bool PreLayerNorm { get; init; } = true;

        /// <summary>
        /// Tie LM head weights to token embedding weights.
        /// Reduces parameters by VocabSize * DModel.
        /// Standard practice for language models.
        /// </summary>
        public bool TieWeights { get; init; } = true;

        // ── GQA ──────────────────────────────────────────────────────────────

        /// <summary>
        /// Number of KV attention heads (Grouped Query Attention).
        /// Must be a divisor of NHeads.
        /// - NKvHeads == NHeads: standard MHA (GPT-1, GPT-2)
        /// - NKvHeads == 1:      MQA (one K/V shared by all Q heads)
        /// - 1 &lt; NKvHeads &lt; NHeads: GQA (Llama 3, Mistral)
        /// Defaults to NHeads (backward-compatible MHA).
        /// </summary>
        public int NKvHeads { get; init; } = 0;  // 0 = same as NHeads

        /// <summary>Resolved KV head count. Returns NKvHeads if set, else NHeads.</summary>
        public int KvHeads => NKvHeads > 0 ? NKvHeads : NHeads;

        // ── RoPE ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Use Rotary Position Embedding (true for Llama/Mistral/Phi/Qwen).
        /// When false, uses absolute positional embeddings (GPT-1, GPT-2).
        /// </summary>
        public bool UseRoPE { get; init; } = false;

        /// <summary>RoPE base frequency theta. Default 10_000 (GPT-NeoX).</summary>
        public float RoPETheta { get; init; } = 10_000f;

        /// <summary>
        /// Optional Llama-3 "llama3" RoPE frequency scaling for long context (the
        /// <c>rope_scaling</c> block in a Llama-3.x config). Null = plain RoPE.
        /// </summary>
        public RopeScaling? RopeScaling { get; init; }

        // ── FFN ───────────────────────────────────────────────────────────────

        /// <summary>
        /// Feed-forward activation function.
        /// GeLU for GPT-2. SwiGLU for Llama/Mistral/Phi/Qwen.
        /// </summary>
        public FeedForwardActivation FfnActivation
        { get; init; } = FeedForwardActivation.GeLU;

        // ── Mixture of Experts (MoE) ──────────────────────────────────────────
        // 0 ⇒ dense FFN (the default for every model loaded today). When > 0 the FFN
        // block is replaced by ExpertCount expert FFNs + a router that activates
        // ExpertUsedCount of them per token (Mixtral / Qwen-MoE style).

        /// <summary>Number of experts in the MoE FFN. 0 ⇒ dense (no MoE).</summary>
        public int ExpertCount { get; init; }

        /// <summary>Experts activated per token (top-k). 0 ⇒ dense; otherwise ≤ <see cref="ExpertCount"/>.</summary>
        public int ExpertUsedCount { get; init; }

        /// <summary>
        /// FFN length of each routed expert (Qwen-MoE: smaller than <see cref="DFF"/>, which is the
        /// shared expert's length). 0 ⇒ fall back to <see cref="DFF"/>.
        /// </summary>
        public int ExpertFeedForwardLength { get; init; }

        /// <summary>True when this model uses a Mixture-of-Experts FFN.</summary>
        public bool IsMixtureOfExperts => ExpertCount > 0 && ExpertUsedCount > 0;

        /// <summary>Total parameter count (weight-tying aware).</summary>
        public long ParameterCount
        {
            get
            {
                var tokEmb = (long)VocabSize * DModel;
                var posEmb = (long)ContextLength * DModel;

                var layerNorm1 = 2L * DModel;

                // MHA now has Q/K/V/output weights plus Q/K/V/output biases:
                // Wq/Wk/Wv/Wo = 4 * DModel * DModel
                // Bq/Bk/Bv/Bo = 4 * DModel
                var attention = 4L * DModel * DModel + 4L * DModel;

                var layerNorm2 = 2L * DModel;
                var feedForward = 2L * DModel * DFF + DFF + DModel;

                var perBlock = layerNorm1 + attention + layerNorm2 + feedForward;
                var finalLN = 2L * DModel;
                var lmHead = TieWeights ? 0 : (long)VocabSize * DModel;

                return tokEmb + posEmb + NLayers * perBlock + finalLN + lmHead;
            }
        }

        public override string ToString() =>
            $"GPT[vocab={VocabSize}, ctx={ContextLength}, d={DModel}, h={NHeads}, L={NLayers}] " +
            $"~{ParameterCount / 1_000_000.0:F0}M params";
    }
}
