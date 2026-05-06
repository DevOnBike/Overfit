// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Zero-copy weight references for one transformer block.
    /// All weights are held as <see cref="TensorStorage{T}"/> references — no data is copied.
    ///
    /// Production constructor takes a <see cref="TransformerBlock"/> directly.
    /// Test constructor uses optional float[] params so callers only specify what they need.
    /// </summary>
    internal readonly struct BlockWeights
    {
        private readonly TensorStorage<float> _ln1Gamma;
        private readonly TensorStorage<float> _ln1Beta;
        private readonly SingleHeadWeights[] _heads;
        private readonly TensorStorage<float> _attentionBias;
        private readonly TensorStorage<float> _ln2Gamma;
        private readonly TensorStorage<float> _ln2Beta;
        private readonly TensorStorage<float> _ffnW1;
        private readonly TensorStorage<float> _ffnB1;
        private readonly TensorStorage<float> _ffnW2;
        private readonly TensorStorage<float> _ffnB2;

        /// <summary>Production constructor — zero-copy references from a real model block.</summary>
        internal BlockWeights(TransformerBlock block, int headCount)
        {
            _ln1Gamma = block.Norm1.Gamma.Data;
            _ln1Beta = block.Norm1.Beta.Data;
            _attentionBias = block.Attention.Bo.Data;
            _ln2Gamma = block.Norm2.Gamma.Data;
            _ln2Beta = block.Norm2.Beta.Data;
            _ffnW1 = block.FFN.W1.Data;
            _ffnB1 = block.FFN.B1.Data;
            _ffnW2 = block.FFN.W2.Data;
            _ffnB2 = block.FFN.B2.Data;

            _heads = new SingleHeadWeights[headCount];
            for (var h = 0; h < headCount; h++)
            {
                _heads[h] = new SingleHeadWeights(block.Attention, h);
            }
        }

        /// <summary>
        /// Test constructor — all params optional, unspecified fields default to empty.
        /// Uses <see cref="SingleHeadWeights"/> instead of raw float[][] for head weights.
        /// </summary>
        internal BlockWeights(
            SingleHeadWeights[]? heads = null,
            float[]? ln1Gamma = null,
            float[]? ln1Beta = null,
            float[]? attentionBias = null,
            float[]? ln2Gamma = null,
            float[]? ln2Beta = null,
            float[]? ffnW1 = null,
            float[]? ffnB1 = null,
            float[]? ffnW2 = null,
            float[]? ffnB2 = null)
        {
            static TensorStorage<float> Store(float[]? a)
                => TensorStorage<float>.FromArray(a ?? Array.Empty<float>());

            _ln1Gamma = Store(ln1Gamma);
            _ln1Beta = Store(ln1Beta);
            _attentionBias = Store(attentionBias);
            _ln2Gamma = Store(ln2Gamma);
            _ln2Beta = Store(ln2Beta);
            _ffnW1 = Store(ffnW1);
            _ffnB1 = Store(ffnB1);
            _ffnW2 = Store(ffnW2);
            _ffnB2 = Store(ffnB2);

            _heads = heads ?? Array.Empty<SingleHeadWeights>();
        }

        public ReadOnlySpan<float> Ln1Gamma => _ln1Gamma.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln1Beta => _ln1Beta.AsReadOnlySpan();
        public ReadOnlySpan<float> AttentionBias => _attentionBias.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln2Gamma => _ln2Gamma.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln2Beta => _ln2Beta.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnW1 => _ffnW1.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnB1 => _ffnB1.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnW2 => _ffnW2.AsReadOnlySpan();
        public ReadOnlySpan<float> FfnB2 => _ffnB2.AsReadOnlySpan();

        public ref readonly SingleHeadWeights Head(int h) => ref _heads[h];
        public int HeadCount => _heads.Length;
    }
}
