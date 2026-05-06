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
        private readonly BlockWeights[]       _blocks;
        private readonly TensorStorage<float> _finalNormGamma;
        private readonly TensorStorage<float> _finalNormBeta;
        private readonly TensorStorage<float>? _lmHead;
        private readonly float[]?             _lmHeadTransposed; // only for TieWeights

        private readonly int _vocabSize;
        private readonly int _dModel;

        internal StackWeights(GPT1Model model)
        {
            var cfg = model.Config;
            _vocabSize = cfg.VocabSize;
            _dModel    = cfg.DModel;

            _blocks = new BlockWeights[cfg.NLayers];
            for (var l = 0; l < cfg.NLayers; l++)
                _blocks[l] = new BlockWeights(model.Blocks[l], cfg.NHeads);

            _finalNormGamma = model.FinalNorm.Gamma.Data;
            _finalNormBeta  = model.FinalNorm.Beta.Data;

            if (cfg.TieWeights)
            {
                // Transpose once at construction — unavoidable for tied weights.
                // GPT-2 uses TieWeights=false so this path is not taken for GPT-2.
                _lmHeadTransposed = BuildTransposedEmbedding(
                model.TokenEmbedding.Weight.Data.AsReadOnlySpan(),
                cfg.VocabSize, cfg.DModel);
            }
            else
            {
                _lmHead = model.LMHead.Data;
            }
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