// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.CodeAnalysis;
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
    [SuppressMessage(
        "IDisposableAnalyzers.Correctness",
        "IDISP008:Don't assign member with injected and created disposables",
        Justification = "Borrowed zero-copy weight handles - this struct never owns the referenced TensorStorage (see type docs).")]
    internal readonly struct SingleHeadWeights
    {
        private static TensorStorage<float> CreateStorage(float[]? a)
        {
            if (a is null || a.Length == 0)
            {
                return new TensorStorage<float>(0);
            }
            var s = new TensorStorage<float>(a.Length);
            a.CopyTo(s.AsSpan());
            return s;
        }

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
            float[] wq,
            float[] wk,
            float[] wv,
            float[] wo,
            float[]? bq = null,
            float[]? bk = null,
            float[]? bv = null)
        {
            _wq = CreateStorage(wq);
            _wk = CreateStorage(wk);
            _wv = CreateStorage(wv);
            _wo = CreateStorage(wo);
            _bq = CreateStorage(bq);
            _bk = CreateStorage(bk);
            _bv = CreateStorage(bv);
        }

        /// <summary>
        /// Zero-copy constructor — binds directly to externally-owned TensorStorage.
        /// Used by <see cref="CachedLlamaInferenceEngine"/> for GQA models.
        /// In GQA mode wk/wv/bk/bv are ignored (supplied via KvHeadWeights instead).
        /// </summary>
        internal SingleHeadWeights(
            TensorStorage<float> wq,
            TensorStorage<float> bq,
            TensorStorage<float> wo,
            TensorStorage<float>? wk = null,
            TensorStorage<float>? bk = null,
            TensorStorage<float>? wv = null,
            TensorStorage<float>? bv = null)
        {
            static TensorStorage<float> Empty() => new(0);
            _wq = wq; _bq = bq; _wo = wo;
            _wk = wk ?? Empty();
            _bk = bk ?? Empty();
            _wv = wv ?? Empty();
            _bv = bv ?? Empty();
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
