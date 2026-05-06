// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Zero-copy weight references for the full GPT stack.
    /// Owns one <see cref="BlockWeights"/> per layer plus final norm and LM head.
    /// </summary>
    internal sealed class StackWeights
    {
        private BlockWeights[]       _blocks = null!;
        private TensorStorage<float> _finalNormGamma = null!;
        private TensorStorage<float> _finalNormBeta  = null!;
        private TensorStorage<float>? _lmHead;
        private float[]?             _lmHeadTransposed;

        private StackWeights() { }

        internal StackWeights(GPT1Model model)
        {
            var cfg = model.Config;

            _blocks = new BlockWeights[cfg.NLayers];
            for (var l = 0; l < cfg.NLayers; l++)
                _blocks[l] = new BlockWeights(model.Blocks[l], cfg.NHeads);

            _finalNormGamma = model.FinalNorm.Gamma.Data;
            _finalNormBeta  = model.FinalNorm.Beta.Data;

            if (cfg.TieWeights)
            {
                // Transpose once at construction — only for tied weights (GPT-1).
                // GPT-2 uses TieWeights=false — this path is not taken for GPT-2.
                _lmHeadTransposed = BuildTransposedEmbedding(
                    model.TokenEmbedding.Weight.Data.AsReadOnlySpan(),
                    cfg.VocabSize,
                    cfg.DModel);
            }
            else
            {
                _lmHead = model.LMHead.Data;
            }
        }

        /// <summary>For unit testing — creates StackWeights from a block factory.</summary>
        internal static StackWeights ForTest(
            int layerCount,
            int headCount,
            Func<int, BlockWeights> blockFactory,
            float[] finalNormGamma,
            float[] finalNormBeta,
            float[] lmHead)
        {
            var sw = new StackWeights
            {
                _blocks           = new BlockWeights[layerCount],
                _finalNormGamma   = TensorStorage<float>.FromArray(finalNormGamma),
                _finalNormBeta    = TensorStorage<float>.FromArray(finalNormBeta),
                _lmHeadTransposed = null,
                _lmHead           = TensorStorage<float>.FromArray(lmHead),
            };

            for (var l = 0; l < layerCount; l++)
                sw._blocks[l] = blockFactory(l);

            return sw;
        }

        public ref readonly BlockWeights Block(int layer) => ref _blocks[layer];
        public int LayerCount                             => _blocks.Length;
        public ReadOnlySpan<float> FinalNormGamma         => _finalNormGamma.AsReadOnlySpan();
        public ReadOnlySpan<float> FinalNormBeta          => _finalNormBeta.AsReadOnlySpan();

        public ReadOnlySpan<float> LmHeadWeights =>
            _lmHeadTransposed is not null
                ? _lmHeadTransposed.AsSpan()
                : _lmHead!.AsReadOnlySpan();

        private static float[] BuildTransposedEmbedding(
            ReadOnlySpan<float> source, int vocabSize, int dModel)
        {
            var buf = new float[dModel * vocabSize];
            for (var t = 0; t < vocabSize; t++)
                for (var d = 0; d < dModel; d++)
                    buf[d * vocabSize + t] = source[t * dModel + d];
            return buf;
        }
    }
}
