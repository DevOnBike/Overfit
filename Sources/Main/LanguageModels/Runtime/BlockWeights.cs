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
        private readonly TensorStorage<float> _qNorm;   // Qwen3 per-head Q RMSNorm (head_dim); empty otherwise
        private readonly TensorStorage<float> _kNorm;   // Qwen3 per-head K RMSNorm (head_dim); empty otherwise
        private readonly TensorStorage<float> _postAttnNorm; // Gemma-2 sandwich norm after attention; empty otherwise
        private readonly TensorStorage<float> _postFfwNorm;  // Gemma-2 sandwich norm after FFN; empty otherwise
        private readonly SingleHeadWeights[] _heads;
        private readonly KvHeadWeights[]? _kvHeads;   // null = MHA (use heads[h].Wk/Wv)

        // Whole-matrix Q4_K attention handles (M2 plumbing for the M3 OVERFIT_REPACK_ATTN decode lever).
        // Empty unless the loader found Q4_K + mmap + repackable Q/K/V/O — then a single repacked 8×8 GEMV
        // per projection replaces the per-head loop (split heads AFTER). All-or-nothing: the M3 path only
        // fires when all four are present (HasWholeAttnQ4K); otherwise decode uses the per-head _heads above.
        private readonly DecodeWeight _wqWhole;
        private readonly DecodeWeight _wkWhole;
        private readonly DecodeWeight _wvWhole;
        private readonly DecodeWeight _woWhole;
        private readonly TensorStorage<float> _attentionBias;
        private readonly TensorStorage<float> _ln2Gamma;
        private readonly TensorStorage<float> _ln2Beta;
        private readonly DecodeWeight _ffnW1;
        private readonly TensorStorage<float> _ffnB1;
        private readonly DecodeWeight _ffnW2;
        private readonly TensorStorage<float> _ffnB2;
        private readonly DecodeWeight _ffnGate;   // SwiGLU Wgate (empty for GeLU)

        // Mixture of Experts (qwen2moe) — all null/default for a dense FFN block.
        private readonly float[]? _moeRouter;          // ffn_gate_inp, input-major [dModel × expertCount]
        private readonly DecodeWeight[]? _moeGate;     // routed experts: gate / up / down
        private readonly DecodeWeight[]? _moeUp;
        private readonly DecodeWeight[]? _moeDown;
        private readonly DecodeWeight _moeSharedGate;  // shared expert SwiGLU
        private readonly DecodeWeight _moeSharedUp;
        private readonly DecodeWeight _moeSharedDown;
        private readonly float[]? _moeSharedGateInp;   // ffn_gate_inp_shexp [dModel] (sigmoid gate)

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
            _qNorm = CreateStorage([]);   // GPT-1/2 blocks have no QK-norm
            _kNorm = CreateStorage([]);
            _postAttnNorm = CreateStorage([]); // GPT-1/2 have no sandwich norm
            _postFfwNorm = CreateStorage([]);
            _kvHeads = null; // MHA — use SingleHeadWeights.Wk/Wv

            _heads = new SingleHeadWeights[headCount];
            for (var h = 0; h < headCount; h++)
            {
                _heads[h] = new SingleHeadWeights(block.Attention, h);
            }

            _moeRouter = null;
            _moeGate = null;
            _moeUp = null;
            _moeDown = null;
            _moeSharedGate = default;
            _moeSharedUp = default;
            _moeSharedDown = default;
            _moeSharedGateInp = null;

            _wqWhole = default;
            _wkWhole = default;
            _wvWhole = default;
            _woWhole = default;
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
            _qNorm = Store(null);
            _kNorm = Store(null);
            _postAttnNorm = Store(null);
            _postFfwNorm = Store(null);

            _heads = heads ?? [];
            _kvHeads = kvHeads;

            _moeRouter = null;
            _moeGate = null;
            _moeUp = null;
            _moeDown = null;
            _moeSharedGate = default;
            _moeSharedUp = default;
            _moeSharedDown = default;
            _moeSharedGateInp = null;

            _wqWhole = default;
            _wkWhole = default;
            _wvWhole = default;
            _woWhole = default;
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
            DecodeWeight ffnW1,
            TensorStorage<float>? ffnB1,
            DecodeWeight ffnW2,
            TensorStorage<float>? ffnB2,
            DecodeWeight ffnGate,
            float[]? moeRouter = null,
            DecodeWeight[]? moeGate = null,
            DecodeWeight[]? moeUp = null,
            DecodeWeight[]? moeDown = null,
            DecodeWeight moeSharedGate = default,
            DecodeWeight moeSharedUp = default,
            DecodeWeight moeSharedDown = default,
            float[]? moeSharedGateInp = null,
            TensorStorage<float>? qNorm = null,
            TensorStorage<float>? kNorm = null,
            TensorStorage<float>? postAttnNorm = null,
            TensorStorage<float>? postFfwNorm = null,
            DecodeWeight wqWhole = default,
            DecodeWeight wkWhole = default,
            DecodeWeight wvWhole = default,
            DecodeWeight woWhole = default)
        {
            static TensorStorage<float> Empty() => TensorStorage<float>.Unpooled(0);

            _qNorm = qNorm ?? Empty();
            _kNorm = kNorm ?? Empty();
            _postAttnNorm = postAttnNorm ?? Empty();
            _postFfwNorm = postFfwNorm ?? Empty();
            _ln1Gamma = ln1Gamma;
            _ln1Beta = ln1Beta ?? Empty();
            _attentionBias = attentionBias ?? Empty();
            _ln2Gamma = ln2Gamma;
            _ln2Beta = ln2Beta ?? Empty();
            _ffnW1 = ffnW1;
            _ffnB1 = ffnB1 ?? Empty();
            _ffnW2 = ffnW2;
            _ffnB2 = ffnB2 ?? Empty();
            _ffnGate = ffnGate;
            _heads = heads ?? [];
            _kvHeads = kvHeads;

            _moeRouter = moeRouter;
            _moeGate = moeGate;
            _moeUp = moeUp;
            _moeDown = moeDown;
            _moeSharedGate = moeSharedGate;
            _moeSharedUp = moeSharedUp;
            _moeSharedDown = moeSharedDown;
            _moeSharedGateInp = moeSharedGateInp;

            _wqWhole = wqWhole;
            _wkWhole = wkWhole;
            _wvWhole = wvWhole;
            _woWhole = woWhole;
        }

        public ReadOnlySpan<float> Ln1Gamma => _ln1Gamma.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln1Beta => _ln1Beta.AsReadOnlySpan();
        public ReadOnlySpan<float> AttentionBias => _attentionBias.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln2Gamma => _ln2Gamma.AsReadOnlySpan();
        public ReadOnlySpan<float> Ln2Beta => _ln2Beta.AsReadOnlySpan();
        public DecodeWeight FfnW1 => _ffnW1;
        public ReadOnlySpan<float> FfnB1 => _ffnB1.AsReadOnlySpan();
        public DecodeWeight FfnW2 => _ffnW2;
        public ReadOnlySpan<float> FfnB2 => _ffnB2.AsReadOnlySpan();

        public ref readonly SingleHeadWeights Head(int h) => ref _heads[h];
        public int HeadCount => _heads.Length;

        /// <summary>Qwen3 per-head QK-RMSNorm present? When true, apply <see cref="QkNormQ"/>/<see cref="QkNormK"/>
        /// to each projected Q/K head (over head_dim) before RoPE.</summary>
        public bool HasQkNorm => _qNorm.AsReadOnlySpan().Length > 0;
        public ReadOnlySpan<float> QkNormQ => _qNorm.AsReadOnlySpan();
        public ReadOnlySpan<float> QkNormK => _kNorm.AsReadOnlySpan();

        /// <summary>Gemma-2 sandwich norm present? When true, RMSNorm the attention/FFN sublayer OUTPUT with
        /// <see cref="PostAttnNorm"/>/<see cref="PostFfwNorm"/> before each residual add.</summary>
        public bool HasPostNorm => _postAttnNorm.AsReadOnlySpan().Length > 0;
        public ReadOnlySpan<float> PostAttnNorm => _postAttnNorm.AsReadOnlySpan();
        public ReadOnlySpan<float> PostFfwNorm => _postFfwNorm.AsReadOnlySpan();

        /// <summary>KV heads for GQA. Null for standard MHA (GPT-1, GPT-2).</summary>
        public bool HasGqa => _kvHeads is not null;
        public int KvHeadCount => _kvHeads?.Length ?? _heads.Length;
        public ref readonly KvHeadWeights KvHead(int kvH) => ref _kvHeads![kvH];

        /// <summary>True when all four whole-matrix Q4_K attention handles are present — the M3
        /// OVERFIT_REPACK_ATTN decode path's gate. Empty on every non-Q4_K / non-mmap / fused-QKV block.</summary>
        public bool HasWholeAttnQ4K =>
            _wqWhole.IsQ4K && _wkWhole.IsQ4K && _wvWhole.IsQ4K && _woWhole.IsQ4K;

        /// <summary>Whole-matrix Q4_K Q projection [nHeads·headDim, dModel]. Valid when <see cref="HasWholeAttnQ4K"/>.</summary>
        public DecodeWeight WqWhole => _wqWhole;
        /// <summary>Whole-matrix Q4_K K projection [nKvHeads·headDim, dModel]. Valid when <see cref="HasWholeAttnQ4K"/>.</summary>
        public DecodeWeight WkWhole => _wkWhole;
        /// <summary>Whole-matrix Q4_K V projection [nKvHeads·headDim, dModel]. Valid when <see cref="HasWholeAttnQ4K"/>.</summary>
        public DecodeWeight WvWhole => _wvWhole;
        /// <summary>Whole-matrix Q4_K O projection [dModel, nHeads·headDim]. Valid when <see cref="HasWholeAttnQ4K"/>.</summary>
        public DecodeWeight WoWhole => _woWhole;

        /// <summary>SwiGLU gate weight. Empty for GeLU/ReLU FFN.</summary>
        public DecodeWeight FfnGate => _ffnGate;

        // ── Mixture of Experts (qwen2moe) ─────────────────────────────────────
        /// <summary>True when this block's FFN is a Mixture-of-Experts.</summary>
        public bool IsMoe => _moeRouter is not null;
        public ReadOnlySpan<float> MoeRouter => _moeRouter;
        public DecodeWeight[] MoeGate => _moeGate!;
        public DecodeWeight[] MoeUp => _moeUp!;
        public DecodeWeight[] MoeDown => _moeDown!;
        public DecodeWeight MoeSharedGate => _moeSharedGate;
        public DecodeWeight MoeSharedUp => _moeSharedUp;
        public DecodeWeight MoeSharedDown => _moeSharedDown;
        public ReadOnlySpan<float> MoeSharedGateInp => _moeSharedGateInp;

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
