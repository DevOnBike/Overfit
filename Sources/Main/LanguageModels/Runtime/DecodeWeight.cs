// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// A decode-time weight matrix, resident in one of four formats:
    /// F32 (input-major <see cref="TensorStorage{T}"/>, the GPT-1/2 path),
    /// Q8_0 (output-major 8-bit — <see cref="Q8Weight"/>),
    /// Q4_K (output-major 4.5-bit super-blocks — <see cref="Q4KWeight"/>), or
    /// Q6_K (output-major 6.5-bit super-blocks — <see cref="Q6KWeight"/>).
    ///
    /// The decode path dispatches once, at the leaf, on the weight's resident
    /// format (<see cref="IsQuantized"/> = Q8, <see cref="IsQ4K"/> = Q4_K,
    /// <see cref="IsQ6K"/> = Q6_K); the rest of the decode stack is
    /// precision-agnostic and just carries this handle through. Implicit
    /// conversions from each backing keep existing construction sites
    /// source-compatible.
    ///
    /// This is the per-type-tagged weight handle — the same scheme ggml uses
    /// (a tensor carries its type, the matmul dispatches a per-type kernel).
    /// </summary>
    public readonly struct DecodeWeight : IDisposable
    {
        private readonly TensorStorage<float>? _f32;
        private readonly Q8Weight? _q8;
        private readonly Q4KWeight? _q4k;
        private readonly Q6KWeight? _q6k;

        public DecodeWeight(TensorStorage<float> f32)
        {
            _f32 = f32 ?? throw new ArgumentNullException(nameof(f32));
            _q8 = null;
            _q4k = null;
            _q6k = null;
        }

        public DecodeWeight(Q8Weight q8)
        {
            _q8 = q8 ?? throw new ArgumentNullException(nameof(q8));
            _f32 = null;
            _q4k = null;
            _q6k = null;
        }

        public DecodeWeight(Q4KWeight q4k)
        {
            _q4k = q4k ?? throw new ArgumentNullException(nameof(q4k));
            _f32 = null;
            _q8 = null;
            _q6k = null;
        }

        public DecodeWeight(Q6KWeight q6k)
        {
            _q6k = q6k ?? throw new ArgumentNullException(nameof(q6k));
            _f32 = null;
            _q8 = null;
            _q4k = null;
        }

        public static implicit operator DecodeWeight(TensorStorage<float> f32) => new(f32);

        public static implicit operator DecodeWeight(Q8Weight q8) => new(q8);

        public static implicit operator DecodeWeight(Q4KWeight q4k) => new(q4k);

        public static implicit operator DecodeWeight(Q6KWeight q6k) => new(q6k);

        /// <summary>True when the weight is resident as Q8_0 specifically.</summary>
        public bool IsQuantized => _q8 is not null;

        /// <summary>True when the weight is resident as Q4_K specifically.</summary>
        public bool IsQ4K => _q4k is not null;

        /// <summary>True when the weight is resident as Q6_K specifically.</summary>
        public bool IsQ6K => _q6k is not null;

        /// <summary>True when no backing is set (default value).</summary>
        public bool IsEmpty => _q8 is null
            && _q4k is null
            && _q6k is null
            && (_f32 is null || _f32.Length == 0);

        /// <summary>Total weight element count, regardless of backing.</summary>
        public long ElementCount => _q6k is not null
            ? (long)_q6k.OutputSize * _q6k.InputSize
            : _q4k is not null
                ? (long)_q4k.OutputSize * _q4k.InputSize
                : _q8 is not null
                    ? (long)_q8.OutputSize * _q8.InputSize
                    : _f32?.Length ?? 0;

        /// <summary>F32 weight span. Valid only when the F32 backing is set.</summary>
        public ReadOnlySpan<float> F32 => _f32!.AsReadOnlySpan();

        /// <summary>Q8_0 weight. Valid only when <see cref="IsQuantized"/> is true.</summary>
        public Q8Weight Quantized => _q8!;

        /// <summary>Q4_K weight. Valid only when <see cref="IsQ4K"/> is true.</summary>
        public Q4KWeight Quantized4K => _q4k!;

        /// <summary>Q6_K weight. Valid only when <see cref="IsQ6K"/> is true.</summary>
        public Q6KWeight Quantized6K => _q6k!;

        /// <summary>
        /// Underlying F32 storage, or null when Q-backed. Used by the in-place
        /// LoRA weight-merge path, which operates only on F32 storage.
        /// </summary>
        public TensorStorage<float>? F32Storage => _f32;

        /// <summary>
        /// Writes output row <paramref name="row"/> as F32 into <paramref name="dst"/>, regardless of
        /// backing — the token-embedding lookup primitive (this weight as a [vocab × dModel] table,
        /// row = token). F32 copies the row; Q4_K/Q6_K/Q8_0 dequantize only that row. Zero-allocation.
        /// For Q-backings the row length is <see cref="ElementCount"/> / outputs; for the F32 backing
        /// the caller's <paramref name="dst"/> length defines the row stride (pass exactly the model dim).
        /// </summary>
        public void DequantizeRow(int row, Span<float> dst)
        {
            if (_q4k is not null) { _q4k.DecodeRow(row, dst); return; }
            if (_q6k is not null) { _q6k.DecodeRow(row, dst); return; }
            if (_q8 is not null) { _q8.DecodeRow(row, dst); return; }
            if (_f32 is not null)
            {
                var rowLength = dst.Length;
                _f32.AsReadOnlySpan().Slice(row * rowLength, rowLength).CopyTo(dst);
                return;
            }

            throw new InvalidOperationException("DequantizeRow on an empty DecodeWeight.");
        }

        /// <summary>Disposes the F32 backing if present; the Q backings own no unmanaged resource.</summary>
        public void Dispose() => _f32?.Dispose();
    }
}
