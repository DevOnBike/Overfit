// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Hyperparameter configuration for GPT-style models.
    ///
    /// GPT-1 (Radford et al., 2018):
    ///   VocabSize     = 40478  (BPE vocabulary)
    ///   ContextLength = 512
    ///   DModel        = 768
    ///   NHeads        = 12
    ///   NLayers       = 12
    ///   DFF           = 3072   (= 4 * DModel)
    ///
    /// GPT-2 Small (for reference):
    ///   VocabSize     = 50257
    ///   ContextLength = 1024
    ///   DModel        = 768
    ///   NHeads        = 12
    ///   NLayers       = 12
    ///   DFF           = 3072
    /// </summary>
    public sealed class GPT1Config
    {
        /// <summary>GPT-1 original configuration.</summary>
        public static readonly GPT1Config GPT1 = new()
        {
            VocabSize     = 40478,
            ContextLength = 512,
            DModel        = 768,
            NHeads        = 12,
            NLayers       = 12,
            DFF           = 3072,
        };

        /// <summary>Small config for testing (fits in memory, fast to instantiate).</summary>
        public static readonly GPT1Config Small = new()
        {
            VocabSize     = 256,
            ContextLength = 16,
            DModel        = 64,
            NHeads        = 4,
            NLayers       = 2,
            DFF           = 256,
        };

        /// <summary>Vocabulary size (number of BPE tokens).</summary>
        public int VocabSize     { get; init; } = 40478;

        /// <summary>Maximum sequence length (context window).</summary>
        public int ContextLength { get; init; } = 512;

        /// <summary>Model dimension (embedding size, d_model).</summary>
        public int DModel        { get; init; } = 768;

        /// <summary>Number of attention heads per block.</summary>
        public int NHeads        { get; init; } = 12;

        /// <summary>Number of Transformer blocks.</summary>
        public int NLayers       { get; init; } = 12;

        /// <summary>Feed-forward inner dimension (typically 4 * DModel).</summary>
        public int DFF           { get; init; } = 3072;

        /// <summary>Layer norm epsilon.</summary>
        public float LNEps       { get; init; } = 1e-5f;

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
        public bool TieWeights   { get; init; } = true;

        /// <summary>Total parameter count (approximate, weight-tying aware).</summary>
        public long ParameterCount
        {
            get
            {
                long tokEmb  = (long)VocabSize * DModel;
                long posEmb  = (long)ContextLength * DModel;
                long perBlock = 2L * DModel                        // LN1
                              + 4L * DModel * DModel + DModel      // MHA
                              + 2L * DModel                        // LN2
                              + 2L * DModel * DFF + DFF + DModel;  // FFN
                long finalLN = 2L * DModel;
                long lmHead  = TieWeights ? 0 : (long)VocabSize * DModel;

                return tokEmb + posEmb + NLayers * perBlock + finalLN + lmHead;
            }
        }

        public override string ToString()
            => $"GPT[vocab={VocabSize}, ctx={ContextLength}, d={DModel}, h={NHeads}, L={NLayers}] " +
               $"~{ParameterCount / 1_000_000.0:F0}M params";
    }
}
