// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Zero-copy weight references for a single attention head.
    ///
    /// Stores <see cref="TensorStorage{T}"/> references directly — no data is copied.
    /// Each .AsReadOnlySpan() call returns a view into the original Parameter storage.
    ///
    /// Lifetime: valid as long as the parent GPT1Model is alive.
    /// LoRA-friendly: if TensorStorage data is updated in-place, spans automatically
    /// reflect the new weights without rebinding.
    /// </summary>
    internal readonly struct SingleHeadWeights
    {
        private readonly TensorStorage<float> _wq;
        private readonly TensorStorage<float> _wk;
        private readonly TensorStorage<float> _wv;
        private readonly TensorStorage<float> _wo;
        private readonly TensorStorage<float> _bq;
        private readonly TensorStorage<float> _bk;
        private readonly TensorStorage<float> _bv;

        internal SingleHeadWeights(MultiHeadAttentionLayer attn, int head)
        {
            _wq = attn.WqHeads[head].Data;
            _wk = attn.WkHeads[head].Data;
            _wv = attn.WvHeads[head].Data;
            _wo = attn.WoHeads[head].Data;
            _bq = attn.BqHeads[head].Data;
            _bk = attn.BkHeads[head].Data;
            _bv = attn.BvHeads[head].Data;
        }

        public ReadOnlySpan<float> Wq => _wq.AsReadOnlySpan();
        public ReadOnlySpan<float> Wk => _wk.AsReadOnlySpan();
        public ReadOnlySpan<float> Wv => _wv.AsReadOnlySpan();
        public ReadOnlySpan<float> Wo => _wo.AsReadOnlySpan();
        public ReadOnlySpan<float> Bq => _bq.AsReadOnlySpan();
        public ReadOnlySpan<float> Bk => _bk.AsReadOnlySpan();
        public ReadOnlySpan<float> Bv => _bv.AsReadOnlySpan();
    }

}
