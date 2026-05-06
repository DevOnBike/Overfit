using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Zero-copy weight references for one transformer block.
    ///
    /// Stores <see cref="TensorStorage{T}"/> references for all weights in the block —
    /// layer norms, per-head attention weights and biases, FFN weights.
    /// No managed heap allocation per inference step.
    /// </summary>
    internal readonly struct BlockWeights
    {
        private readonly TensorStorage<float> _ln1Gamma;
        private readonly TensorStorage<float> _ln1Beta;
        private readonly SingleHeadWeights[]  _heads;
        private readonly TensorStorage<float> _attentionBias;
        private readonly TensorStorage<float> _ln2Gamma;
        private readonly TensorStorage<float> _ln2Beta;
        private readonly TensorStorage<float> _ffnW1;
        private readonly TensorStorage<float> _ffnB1;
        private readonly TensorStorage<float> _ffnW2;
        private readonly TensorStorage<float> _ffnB2;

        internal BlockWeights(TransformerBlock block, int headCount)
        {
            _ln1Gamma      = block.Norm1.Gamma.Data;
            _ln1Beta       = block.Norm1.Beta.Data;
            _attentionBias = block.Attention.Bo.Data;
            _ln2Gamma      = block.Norm2.Gamma.Data;
            _ln2Beta       = block.Norm2.Beta.Data;
            _ffnW1         = block.FFN.W1.Data;
            _ffnB1         = block.FFN.B1.Data;
            _ffnW2         = block.FFN.W2.Data;
            _ffnB2         = block.FFN.B2.Data;

            _heads = new SingleHeadWeights[headCount];
            for (var h = 0; h < headCount; h++)
                _heads[h] = new SingleHeadWeights(block.Attention, h);
        }

        public ReadOnlySpan<float> Ln1Gamma      => _ln1Gamma.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln1Beta        => _ln1Beta.AsReadOnlySpan();
        public ReadOnlySpan<float> AttentionBias  => _attentionBias.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln2Gamma       => _ln2Gamma.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln2Beta        => _ln2Beta.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnW1          => _ffnW1.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnB1          => _ffnB1.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnW2          => _ffnW2.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnB2          => _ffnB2.AsReadOnlySpan();

        public ref readonly SingleHeadWeights Head(int h) => ref _heads[h];
    }
}