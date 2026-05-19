// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// A decode-time weight matrix, resident either as F32 (input-major
    /// <see cref="TensorStorage{T}"/> — the current path) or as Q8_0
    /// (output-major 8-bit quantized — see <see cref="Q8Weight"/>).
    ///
    /// The decode kernel dispatches once, at the leaf, on <see cref="IsQuantized"/>;
    /// the rest of the decode stack is precision-agnostic and just carries this
    /// value through. Implicit conversions from <see cref="TensorStorage{T}"/>
    /// and <see cref="Q8Weight"/> keep every existing F32 construction site
    /// (GPT-1/2, the binary loader, tests) source-compatible.
    ///
    /// This is the per-type-tagged weight handle — the same scheme ggml uses
    /// (a tensor carries its type, the matmul dispatches a per-type kernel).
    /// </summary>
    public readonly struct DecodeWeight : IDisposable
    {
        private readonly TensorStorage<float>? _f32;
        private readonly Q8Weight? _q8;

        public DecodeWeight(TensorStorage<float> f32)
        {
            _f32 = f32 ?? throw new ArgumentNullException(nameof(f32));
            _q8 = null;
        }

        public DecodeWeight(Q8Weight q8)
        {
            _q8 = q8 ?? throw new ArgumentNullException(nameof(q8));
            _f32 = null;
        }

        public static implicit operator DecodeWeight(TensorStorage<float> f32) => new(f32);

        public static implicit operator DecodeWeight(Q8Weight q8) => new(q8);

        /// <summary>True when the weight is resident as Q8_0.</summary>
        public bool IsQuantized => _q8 is not null;

        /// <summary>True when neither backing is set (default value).</summary>
        public bool IsEmpty => _f32 is null && _q8 is null;

        /// <summary>F32 weight span. Valid only when <see cref="IsQuantized"/> is false.</summary>
        public ReadOnlySpan<float> F32 => _f32!.AsReadOnlySpan();

        /// <summary>Q8_0 weight. Valid only when <see cref="IsQuantized"/> is true.</summary>
        public Q8Weight Quantized => _q8!;

        /// <summary>
        /// Underlying F32 storage, or null when Q8-backed. Used by the in-place
        /// LoRA weight-merge path, which operates only on F32 storage.
        /// </summary>
        public TensorStorage<float>? F32Storage => _f32;

        /// <summary>Disposes the F32 backing if present; the Q8 backing owns no unmanaged resource.</summary>
        public void Dispose() => _f32?.Dispose();
    }
}
