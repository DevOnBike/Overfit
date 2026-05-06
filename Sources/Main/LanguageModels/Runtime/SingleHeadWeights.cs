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
    /// Holds <see cref="TensorStorage{T}"/> references — no data copied.
    /// Lifetime: valid as long as the parent GPT1Model is alive.
    /// LoRA-friendly: TensorStorage updates are visible immediately.
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

        /// <summary>Production constructor — zero-copy references from a real attention layer.</summary>
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

        /// <summary>
        /// Test constructor — unspecified biases default to empty.
        /// </summary>
        internal SingleHeadWeights(
            float[]  wq,
            float[]  wk,
            float[]  wv,
            float[]  wo,
            float[]? bq = null,
            float[]? bk = null,
            float[]? bv = null)
        {
            static TensorStorage<float> Store(float[]? a)
                => TensorStorage<float>.FromArray(a ?? Array.Empty<float>());

            _wq = Store(wq);
            _wk = Store(wk);
            _wv = Store(wv);
            _wo = Store(wo);
            _bq = Store(bq);
            _bk = Store(bk);
            _bv = Store(bv);
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
