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
    /// Zero-copy weight references for one transformer block.
    /// All weights are held as <see cref="TensorStorage{T}"/> references — no data is copied.
    ///
    /// Production constructor takes a <see cref="TransformerBlock"/> directly.
    /// Test constructor uses optional float[] params so callers only specify what they need.
    /// </summary>
    [SuppressMessage(
        "IDisposableAnalyzers.Correctness",
        "IDISP008:Don't assign member with injected and created disposables",
        Justification = "Borrowed zero-copy weight handles - this struct never owns the referenced TensorStorage (see type docs).")]
    internal readonly struct BlockWeights
    {
        private readonly TensorStorage<float> _ln1Gamma;
        private readonly TensorStorage<float> _ln1Beta;
        private readonly SingleHeadWeights[] _heads;
        private readonly KvHeadWeights[]? _kvHeads;   // null = MHA (use heads[h].Wk/Wv)
        private readonly TensorStorage<float> _attentionBias;
        private readonly TensorStorage<float> _ln2Gamma;
        private readonly TensorStorage<float> _ln2Beta;
        private readonly TensorStorage<float> _ffnW1;
        private readonly TensorStorage<float> _ffnB1;
        private readonly TensorStorage<float> _ffnW2;
        private readonly TensorStorage<float> _ffnB2;
        private readonly TensorStorage<float> _ffnGate;   // SwiGLU Wgate (empty for GeLU)

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
            _ffnGate = CreateStorage([]); // GeLU — no gate
            _kvHeads = null; // MHA — use SingleHeadWeights.Wk/Wv

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
            KvHeadWeights[]? kvHeads = null,
            float[]? ln1Gamma = null,
            float[]? ln1Beta = null,
            float[]? attentionBias = null,
            float[]? ln2Gamma = null,
            float[]? ln2Beta = null,
            float[]? ffnW1 = null,
            float[]? ffnB1 = null,
            float[]? ffnW2 = null,
            float[]? ffnB2 = null,
            float[]? ffnGate = null)
        {
            static TensorStorage<float> Store(float[]? a)
                => CreateStorage(a ?? []);

            _ln1Gamma = Store(ln1Gamma);
            _ln1Beta = Store(ln1Beta);
            _attentionBias = Store(attentionBias);
            _ln2Gamma = Store(ln2Gamma);
            _ln2Beta = Store(ln2Beta);
            _ffnW1 = Store(ffnW1);
            _ffnB1 = Store(ffnB1);
            _ffnW2 = Store(ffnW2);
            _ffnB2 = Store(ffnB2);
            _ffnGate = Store(ffnGate);

            _heads = heads ?? [];
            _kvHeads = kvHeads;
        }

        /// <summary>
        /// Zero-copy production constructor — binds directly to externally-owned TensorStorages.
        /// No data is duplicated; the weights live in the loader's allocations.
        /// Pass null for optional fields (Ln1Beta/Ln2Beta/biases) and an empty-storage sentinel
        /// will be wired in their place.
        /// </summary>
        internal BlockWeights(
            SingleHeadWeights[] heads,
            KvHeadWeights[]? kvHeads,
            TensorStorage<float> ln1Gamma,
            TensorStorage<float>? ln1Beta,
            TensorStorage<float>? attentionBias,
            TensorStorage<float> ln2Gamma,
            TensorStorage<float>? ln2Beta,
            TensorStorage<float> ffnW1,
            TensorStorage<float>? ffnB1,
            TensorStorage<float> ffnW2,
            TensorStorage<float>? ffnB2,
            TensorStorage<float>? ffnGate)
        {
            static TensorStorage<float> Empty() => TensorStorage<float>.Unpooled(0);

            _ln1Gamma = ln1Gamma;
            _ln1Beta = ln1Beta ?? Empty();
            _attentionBias = attentionBias ?? Empty();
            _ln2Gamma = ln2Gamma;
            _ln2Beta = ln2Beta ?? Empty();
            _ffnW1 = ffnW1;
            _ffnB1 = ffnB1 ?? Empty();
            _ffnW2 = ffnW2;
            _ffnB2 = ffnB2 ?? Empty();
            _ffnGate = ffnGate ?? Empty();
            _heads = heads ?? [];
            _kvHeads = kvHeads;
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

        /// <summary>KV heads for GQA. Null for standard MHA (GPT-1, GPT-2).</summary>
        public bool HasGqa => _kvHeads is not null;
        public int KvHeadCount => _kvHeads?.Length ?? _heads.Length;
        public ref readonly KvHeadWeights KvHead(int kvH) => ref _kvHeads![kvH];

        /// <summary>SwiGLU gate weight. Empty for GeLU/ReLU FFN.</summary>
        public ReadOnlySpan<float> FfnGate => _ffnGate.AsReadOnlySpan();

        private static TensorStorage<float> CreateStorage(float[]? source)
        {
            if (source is null || source.Length == 0)
            {
                return new TensorStorage<float>(0);
            }

            var storage = new TensorStorage<float>(source.Length);

            source.CopyTo(storage.AsSpan());
            return storage;
        }

    }
}
