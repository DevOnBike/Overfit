// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Zero-copy weight references for one KV attention head.
    ///
    /// In GQA (Grouped Query Attention) a single KvHeadWeights is shared
    /// by <c>nQHeads / nKvHeads</c> query heads.
    ///
    /// For standard MHA (GPT-1, GPT-2) KvHeadWeights are embedded inside
    /// SingleHeadWeights — this struct is only used when nKvHeads &lt; nQHeads.
    /// </summary>
    internal readonly struct KvHeadWeights
    {
        private readonly TensorStorage<float> _wk;
        private readonly TensorStorage<float> _wv;
        private readonly TensorStorage<float> _bk;
        private readonly TensorStorage<float> _bv;

        /// <summary>
        /// Constructs from raw arrays — used for testing and for future Llama model binding.
        /// Unspecified biases default to empty (Llama has no attention biases).
        /// </summary>
        internal KvHeadWeights(
            float[]  wk,
            float[]  wv,
            float[]? bk = null,
            float[]? bv = null)
        {
            static TensorStorage<float> Store(float[]? a)
                => TensorStorage<float>.FromArray(a ?? Array.Empty<float>());

            _wk = Store(wk);
            _wv = Store(wv);
            _bk = Store(bk);
            _bv = Store(bv);
        }

        /// <summary>
        /// Internal constructor — binds directly to externally-owned TensorStorage.
        /// Used by the Llama model adapter (zero-copy, no data allocation).
        /// </summary>
        internal KvHeadWeights(
            TensorStorage<float> wk,
            TensorStorage<float> wv,
            TensorStorage<float> bk,
            TensorStorage<float> bv)
        {
            _wk = wk;
            _wv = wv;
            _bk = bk;
            _bv = bv;
        }

        public ReadOnlySpan<float> Wk => _wk.AsReadOnlySpan();
        public ReadOnlySpan<float> Wv => _wv.AsReadOnlySpan();
        public ReadOnlySpan<float> Bk => _bk.AsReadOnlySpan();
        public ReadOnlySpan<float> Bv => _bv.AsReadOnlySpan();
    }
}
